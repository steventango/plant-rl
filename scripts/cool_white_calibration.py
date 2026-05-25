"""
Interactive script to measure and record cool_white PPFD calibration data.

Usage:
    python scripts/cool_white_calibration.py --zone 1

For each action value from 0.80 to 1.50 (0.05 steps), the script sends a
raw action to the lightbar with only the cool_white channel active, then
prompts for the measured PPFD value. Results are written to the zone JSON
immediately after each measurement so partial runs are preserved.
"""

import argparse
import json
from pathlib import Path

import requests

CONFIG_DIR = Path(__file__).parent.parent / "src" / "environments" / "PlantGrowthChamber" / "configs"
ACTION_VALUES = [round(0.80 + 0.05 * i, 2) for i in range(15)]  # 0.80 … 1.50


def send_action(lightbar_url: str, cool_white_value: float) -> None:
    action = [[0.0, cool_white_value, 0.0, 0.0, 0.0, 0.0]] * 2
    requests.put(lightbar_url, json={"array": action}, timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", type=int, required=True, help="Zone number (e.g. 1 for alliance-zone01)")
    args = parser.parse_args()

    config_path = CONFIG_DIR / f"alliance-zone{args.zone:02d}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")

    data = json.loads(config_path.read_text())
    cal = data["zone"]["calibration"]
    lightbar_url = data["zone"]["lightbar_url"]

    if lightbar_url is None:
        raise ValueError("Zone has no lightbar_url configured.")

    print(f"Zone: alliance-zone{args.zone:02d}")
    print(f"Lightbar: {lightbar_url}")
    print(f"Measuring cool_white PPFD at action values: {ACTION_VALUES}\n")

    for action_val in ACTION_VALUES:
        # Find the index for this action value in the calibration array
        try:
            idx = next(i for i, a in enumerate(cal["action"]) if round(a, 2) == action_val)
        except StopIteration:
            print(f"  action={action_val} not found in calibration array — skipping")
            continue

        current = cal["cool_white"][idx]
        print(f"action={action_val:.2f}  (current cool_white={current})")

        try:
            send_action(lightbar_url, action_val)
        except Exception as e:
            print(f"  Warning: failed to send action — {e}")

        raw = input("  Measured PPFD (Enter to keep existing): ").strip()
        if raw:
            cal["cool_white"][idx] = float(raw)
            config_path.write_text(json.dumps(data, indent=4))
            print(f"  Saved {float(raw)}")
        else:
            print(f"  Kept {current}")

    # Turn lights off
    try:
        send_action(lightbar_url, 0.0)
        print("\nLights off.")
    except Exception as e:
        print(f"\nWarning: could not turn lights off — {e}")

    print("Done.")


if __name__ == "__main__":
    main()
