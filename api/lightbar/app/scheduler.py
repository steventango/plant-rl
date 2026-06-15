"""On-device fallback lighting schedule.

The control host normally drives the lightbar every minute via PUT /action. When
the network between the host and this Pi drops, those calls stop arriving and the
photoperiod transitions are missed. This module lets the device enforce a small
time-of-day schedule on its own so the lights still turn on/off on time without
the network.

Behaviour is *transition-only*: an entry is applied once when its HH:MM is
crossed, and the device holds that action until the next entry. It does not
re-assert between transitions, so live PUT /action calls take over seamlessly
during normal operation.

The container clock is UTC, so a schedule always carries an explicit timezone and
all comparisons happen in that timezone.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)

# compose.yml bind-mounts .:/app, so this file survives `docker compose restart`.
SCHEDULE_PATH = Path("/app/schedule.json")
TICK_SECONDS = 20


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    hour, minute = int(hour), int(minute)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"time out of range: {value}")
    return hour, minute


def _fire_datetime_at_or_before(entry: dict, ref_dt: datetime) -> datetime:
    """Most recent datetime <= ref_dt whose time-of-day matches entry['time'].

    Anchoring each entry to an actual date (today if its HH:MM has already passed,
    otherwise yesterday) makes both helpers below handle the midnight wrap without
    special cases.
    """
    hour, minute = _parse_hhmm(entry["time"])
    # ref_dt is tz-aware; .replace keeps the ZoneInfo, which recomputes the UTC
    # offset for the new wall time, so the instant comparisons in the callers are
    # correct across DST. (A wall time inside a DST gap/overlap resolves to
    # ZoneInfo's default fold, which is fine for firing a transition.)
    candidate = ref_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate > ref_dt:
        candidate -= timedelta(days=1)
    return candidate


def active_entry(entries: list[dict], now_dt: datetime) -> dict | None:
    """Return the entry currently in effect (most recent at-or-before now_dt).

    Wraps past midnight; returns None for an empty schedule. Used to restore the
    correct state when a schedule is (re)set or loaded after a container restart.
    """
    if not entries:
        return None
    return max(entries, key=lambda e: _fire_datetime_at_or_before(e, now_dt))


def entries_due_between(
    entries: list[dict], last_dt: datetime | None, now_dt: datetime
) -> list[dict]:
    """Entries whose HH:MM falls in (last_dt, now_dt], in fire order.

    Assumes now_dt - last_dt < 24h (true for the short tick), so each entry fires
    at most once per window. Handles the midnight wrap via the date anchoring in
    _fire_datetime_at_or_before.
    """
    if not entries or last_dt is None or now_dt <= last_dt:
        return []
    due = []
    for entry in entries:
        fire = _fire_datetime_at_or_before(entry, now_dt)
        if last_dt < fire <= now_dt:
            due.append((fire, entry))
    due.sort(key=lambda item: item[0])
    return [entry for _, entry in due]


class Scheduler:
    """Holds the current schedule and applies it on a background daemon thread."""

    def __init__(self, get_lightbar, schedule_path=SCHEDULE_PATH, tick_seconds=TICK_SECONDS):
        self._get_lightbar = get_lightbar
        self._schedule_path = Path(schedule_path)
        self._tick_seconds = tick_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.timezone: str | None = None
        self.entries: list[dict] = []
        self.last_check: datetime | None = None
        self.last_fired: datetime | None = None
        # When True, the next tick reapplies the currently-active entry instead of
        # only firing crossed transitions, so the device snaps to the correct state
        # after a reboot mid-outage. Armed only by load() (restart recovery), NOT by
        # a live set(): overriding the action a connected agent just chose would be
        # wrong; the next transition handles a live set going forward.
        self._reapply_pending = False

    # -- schedule management --
    def set(self, timezone: str, entries: list[dict], reapply: bool = False) -> None:
        """Replace the schedule (clears old + sets new) and persist it.

        ``reapply`` arms an immediate snap to the currently-active entry on the next
        tick; reserved for load()/restart recovery, off for live updates.
        """
        ZoneInfo(timezone)  # validate; raises on unknown tz
        for entry in entries:
            _parse_hhmm(entry["time"])  # validate; raises on bad time
        with self._lock:
            self.timezone = timezone
            self.entries = entries
            self.last_fired = None
            self._reapply_pending = reapply
            self._persist_locked()

    def clear(self) -> None:
        with self._lock:
            self.timezone = None
            self.entries = []
            self.last_check = None
            self.last_fired = None
            self._reapply_pending = False
            try:
                self._schedule_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("failed to remove schedule file")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "timezone": self.timezone,
                "entries": self.entries,
                "last_check": self.last_check.isoformat() if self.last_check else None,
                "last_fired": self.last_fired.isoformat() if self.last_fired else None,
            }

    # -- persistence --
    def _persist_locked(self) -> None:
        # Write to a sibling temp file then atomically rename, so a crash mid-write
        # can't leave a truncated schedule.json that fails to parse on restart.
        try:
            tmp = self._schedule_path.with_name(self._schedule_path.name + ".tmp")
            tmp.write_text(
                json.dumps({"timezone": self.timezone, "entries": self.entries})
            )
            os.replace(tmp, self._schedule_path)
        except OSError:
            logger.exception("failed to persist schedule file")

    def load(self) -> None:
        try:
            data = json.loads(self._schedule_path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.exception("failed to load schedule file")
            return
        timezone = data.get("timezone")
        entries = data.get("entries") or []
        if timezone and entries:
            try:
                self.set(timezone, entries, reapply=True)
            except Exception:
                logger.exception("failed to apply loaded schedule")

    # -- background loop --
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="lightbar-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("scheduler tick failed")
            self._stop.wait(self._tick_seconds)

    def _tick(self, now: datetime | None = None) -> None:
        with self._lock:
            timezone = self.timezone
            entries = list(self.entries)
            last_check = self.last_check
            # Consume the reapply flag under the same lock as the read, so a set()
            # that lands while this tick runs re-arms it for the next tick rather
            # than having its request cleared by our tail update.
            reapply = self._reapply_pending
            self._reapply_pending = False
        if not timezone or not entries:
            return
        if now is None:
            now = datetime.now(ZoneInfo(timezone))

        if reapply:
            entry = active_entry(entries, now)
            to_fire = [entry] if entry is not None else []
        else:
            to_fire = entries_due_between(entries, last_check, now)

        for entry in to_fire:
            self._apply(entry)

        with self._lock:
            self.last_check = now
            if to_fire:
                self.last_fired = now

    def _apply(self, entry: dict) -> None:
        try:
            self._get_lightbar().step(np.array(entry["action"]))
            logger.info("scheduler applied entry time=%s", entry.get("time"))
        except Exception:
            logger.exception("scheduler failed to apply entry time=%s", entry.get("time"))
