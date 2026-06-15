from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from ..app.scheduler import Scheduler, active_entry, entries_due_between

TZ = "Etc/GMT-2"

ON = {"time": "08:59", "action": (np.ones((2, 6)) * 0.1).tolist()}
OFF = {"time": "21:00", "action": np.zeros((2, 6)).tolist()}
ENTRIES = [ON, OFF]


def dt(hour, minute, day=1):
    return datetime(2026, 6, day, hour, minute, tzinfo=ZoneInfo(TZ))


class RecorderLightbar:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).tolist())


def test_active_entry():
    assert active_entry(ENTRIES, dt(10, 0)) is ON
    assert active_entry(ENTRIES, dt(22, 0)) is OFF
    # Before the first ON of the day -> previous day's OFF is still in effect.
    assert active_entry(ENTRIES, dt(3, 0)) is OFF
    assert active_entry([], dt(10, 0)) is None


def test_entries_due_between_fires_once_on_crossing():
    assert entries_due_between(ENTRIES, dt(8, 58), dt(9, 0)) == [ON]
    assert entries_due_between(ENTRIES, dt(20, 59), dt(21, 1)) == [OFF]


def test_entries_due_between_excludes_already_fired_boundary():
    # The interval is half-open (last, now]; an entry exactly at last already fired.
    assert entries_due_between(ENTRIES, dt(8, 59), dt(9, 0)) == []


def test_entries_due_between_no_crossing():
    assert entries_due_between(ENTRIES, dt(10, 0), dt(10, 5)) == []


def test_entries_due_between_midnight_wrap():
    midnight = [{"time": "00:00", "action": np.zeros((2, 6)).tolist()}]
    due = entries_due_between(midnight, dt(23, 50, day=1), dt(0, 10, day=2))
    assert due == midnight


def test_entries_due_between_handles_missing_last():
    assert entries_due_between(ENTRIES, None, dt(10, 0)) == []


def test_set_does_not_reapply_active_entry(tmp_path):
    # A live set() must not override the action a connected agent just chose;
    # only transitions fire going forward.
    recorder = RecorderLightbar()
    scheduler = Scheduler(lambda: recorder, schedule_path=tmp_path / "schedule.json")
    scheduler.set(TZ, ENTRIES)

    scheduler._tick(now=dt(10, 0))

    assert recorder.actions == []
    assert scheduler.snapshot()["timezone"] == TZ


def test_load_reapplies_active_entry_on_first_tick(tmp_path):
    # Restart recovery: a loaded schedule snaps to the currently-active entry.
    path = tmp_path / "schedule.json"
    Scheduler(lambda: RecorderLightbar(), schedule_path=path).set(TZ, ENTRIES)

    recorder = RecorderLightbar()
    scheduler = Scheduler(lambda: recorder, schedule_path=path)
    scheduler.load()

    scheduler._tick(now=dt(10, 0))

    assert recorder.actions == [ON["action"]]
    assert scheduler.snapshot()["last_fired"] is not None


def test_transition_fires_without_reapply(tmp_path):
    recorder = RecorderLightbar()
    scheduler = Scheduler(lambda: recorder, schedule_path=tmp_path / "schedule.json")
    scheduler.set(TZ, ENTRIES)

    scheduler._tick(now=dt(10, 0))  # establishes last_check, no reapply
    scheduler._tick(now=dt(21, 1))  # crosses 21:00 -> OFF

    assert recorder.actions == [OFF["action"]]


def test_clear(tmp_path):
    path = tmp_path / "schedule.json"
    scheduler = Scheduler(lambda: RecorderLightbar(), schedule_path=path)
    scheduler.set(TZ, ENTRIES)
    assert path.exists()

    scheduler.clear()

    assert scheduler.snapshot()["timezone"] is None
    assert scheduler.snapshot()["entries"] == []
    assert not path.exists()


def test_persistence_survives_reload(tmp_path):
    path = tmp_path / "schedule.json"
    Scheduler(lambda: RecorderLightbar(), schedule_path=path).set(TZ, ENTRIES)

    reloaded = Scheduler(lambda: RecorderLightbar(), schedule_path=path)
    reloaded.load()

    snapshot = reloaded.snapshot()
    assert snapshot["timezone"] == TZ
    assert snapshot["entries"] == ENTRIES
