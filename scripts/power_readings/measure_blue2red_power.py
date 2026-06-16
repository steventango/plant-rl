"""
Sweep the lightbar through 21 color steps (BLUE → BALANCED → RED) for each
alliance zone and append smartplug power readings to a CSV file.

Usage (from repo root, with KASA_USERNAME/KASA_PASSWORD set in .env):
    uv run python scripts/power_readings/measure_blue2red_power.py
    uv run python scripts/power_readings/measure_blue2red_power.py --zones 1,2 --wait 5 --reads 3
    uv run python scripts/power_readings/measure_blue2red_power.py --dry-run

Output: scripts/power_readings/blue2red_power.csv
    Columns: timestamp, zone, scalar_action, power_w
    New runs append rows; existing data is never overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from environments.PlantGrowthChamber.SmartPlugClient import SmartPlugClient
from environments.PlantGrowthChamber.specs.actions import ContinuousColorAction
from environments.PlantGrowthChamber.zones import (
    ZONE_IDENTIFIERS,
    load_zone_from_config,
)

CSV_PATH = Path(__file__).parent / "blue2red_power.csv"
CSV_HEADER = ["timestamp", "zone", "scalar_action", "power_w"]

_spec = ContinuousColorAction()
SCALAR_ACTIONS = np.linspace(-1.0, 1.0, 21)
ACTIONS = [_spec.decode(a) for a in SCALAR_ACTIONS]  # 21 × 6-dim vectors


async def set_lightbar(
    session: aiohttp.ClientSession, url: str, action: np.ndarray
) -> None:
    calibrated = np.clip(action, None, 1.0)
    tiled = np.tile(calibrated, (2, 1))
    async with session.put(url, json={"array": tiled.tolist()}) as resp:
        resp.raise_for_status()


async def read_power_avg(
    client: SmartPlugClient, host: str, n: int, interval: float
) -> float | None:
    readings: list[float] = []
    for i in range(n):
        result = await client.read(host)
        if result is not None and result.get("power") is not None:
            readings.append(result["power"])
        if i < n - 1:
            await asyncio.sleep(interval)
    return float(np.mean(readings)) if readings else None


async def measure_zone(
    zone_id: str,
    client: SmartPlugClient,
    session: aiohttp.ClientSession,
    wait: float,
    reads: int,
    dry_run: bool,
) -> list[float | None]:
    zone = load_zone_from_config(zone_id)
    if zone.lightbar_url is None:
        print(f"  [{zone_id}] No lightbar_url — skipping")
        return [None] * 21
    if zone.smart_plug_host is None:
        print(f"  [{zone_id}] No smart_plug_host — skipping")
        return [None] * 21

    power_readings: list[float | None] = []

    for i, raw_action in enumerate(ACTIONS):
        scalar = float(SCALAR_ACTIONS[i])
        label = f"a={scalar:+.1f}"
        action = (
            zone.calibration.get_calibrated_action(raw_action)
            if zone.calibration
            else raw_action
        )

        if dry_run:
            print(f"  [{zone_id}] {label}  action={np.round(action, 4).tolist()}")
            power_readings.append(None)
            continue

        try:
            await set_lightbar(session, zone.lightbar_url, action)
        except Exception as e:
            print(f"  [{zone_id}] {label}  lightbar error: {e}")
            power_readings.append(None)
            continue

        await asyncio.sleep(wait)

        power = await read_power_avg(client, zone.smart_plug_host, reads, interval=1.0)
        power_readings.append(power)
        print(f"  [{zone_id}] {label}  power={power} W")

    return power_readings


def append_to_csv(
    timestamp: str, zone_id: str, power_readings: list[float | None]
) -> None:
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for scalar, power in zip(SCALAR_ACTIONS, power_readings):
            writer.writerow([timestamp, zone_id, f"{scalar:.1f}", power])
    print(f"  [{zone_id}] Appended to {CSV_PATH}")


async def main(zone_ids: list[str], wait: float, reads: int, dry_run: bool) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    client = SmartPlugClient()
    timeout = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for zone_id in zone_ids:
            print(f"\n=== {zone_id} ===")
            power_readings = await measure_zone(
                zone_id, client, session, wait, reads, dry_run
            )
            if not dry_run:
                append_to_csv(timestamp, zone_id, power_readings)

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--zones",
        default=None,
        help="Comma-separated zone numbers to run, e.g. 1,3,5 (default: all 1–12)",
    )
    p.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Seconds to wait after setting light (default: 5)",
    )
    p.add_argument(
        "--reads",
        type=int,
        default=3,
        help="Power readings to average per step (default: 3)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print actions without touching hardware"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    zone_ids = (
        [f"alliance-zone{int(z):02d}" for z in args.zones.split(",")]
        if args.zones is not None
        else ZONE_IDENTIFIERS
    )
    asyncio.run(main(zone_ids, args.wait, args.reads, args.dry_run))
