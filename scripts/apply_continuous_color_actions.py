"""Sweep a zone's lightbar through a fixed set of ContinuousColorAction values."""

import argparse
import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from environments.PlantGrowthChamber.specs.actions import ContinuousColorAction

ACTIONS = [-1, -3/4, -2/4, -1/4, 0, 1/4, 2/4, 3/4, 1]


async def main(zone: str, interval: float):
    chamber = PlantGrowthChamber(zone=zone)
    spec = ContinuousColorAction()
    try:
        for value in ACTIONS:
            action = spec.decode(value)
            print(f"Applying action {value:+.3f} -> {action}")
            await chamber.put_action(action)
            await asyncio.sleep(interval)
    finally:
        await chamber.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", required=True, help="Zone identifier, e.g. alliance-zone01")
    parser.add_argument(
        "--interval", type=float, default=3.0, help="Seconds to wait between actions"
    )
    args = parser.parse_args()
    asyncio.run(main(args.zone, args.interval))
