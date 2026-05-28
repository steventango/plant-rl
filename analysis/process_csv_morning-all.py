import argparse
import glob
import os
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

FOLDER_PATHS = [
    "/data/plant-rl/online/E17/P1/",
    "/data/plant-rl/online/E17/P2",
]

ACTION_COLS = [f"action.{i}" for i in range(6)]

_AGENT_ACTION_MAP = [
    (-1, [59.5, 38.06567251461989, 4.161520467836257, 0.0, 3.2728070175438595, 0.0]),
    (1, [11.841678939617085, 43.43770741286205, 4.748816887579775, 0.0, 46.15, 0.0]),
]

ACTION_TOL = 0.01

MORNING_HOUR = 9
MORNING_MINUTE = 30


def decode_agent_action(action_vec):
    for scalar, expected in _AGENT_ACTION_MAP:
        if np.allclose(action_vec, expected, atol=ACTION_TOL):
            return scalar
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", type=int, required=True, help="Zone ID (e.g. 1)")
    args = parser.parse_args()

    zone_id = args.zone
    zone_str = f"alliance-zone{zone_id:02d}"
    output_path = ROOT / f"analysis/data/E17_zone{zone_id:02d}_morning.csv"

    csv_files = []
    for fp in FOLDER_PATHS:
        pattern = os.path.join(fp, "*", zone_str, "raw_*.csv")
        csv_files.extend(sorted(glob.glob(pattern)))

    print(f"Found {len(csv_files)} CSV files for zone {zone_id}:")
    for f in csv_files:
        print(" ", f)

    rows = []
    counter = 0
    for fpath in csv_files:
        df = pd.read_csv(fpath)

        tz = df["timezone"].dropna().iloc[0]
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["local_time"] = df["time"].dt.tz_convert(tz)

        mask = (
            df["plant_id"].notna()
            & (df["local_time"].dt.hour == MORNING_HOUR)
            & (df["local_time"].dt.minute == MORNING_MINUTE)
        )
        df = df[mask]
        if "area" not in df.columns:
            print(f"  Skipping {fpath}: missing 'area' column")
            counter += 1
            continue
        df = df[df["area"] > 0]

        for _, plant_row in df.iterrows():
            action_vec = [plant_row[col] for col in ACTION_COLS]
            rows.append(
                {
                    "time": plant_row["local_time"],
                    "plant_id": plant_row["plant_id"],
                    "area": plant_row["area"],
                    "solidity": plant_row["solidity"],
                    "agent_action": decode_agent_action(action_vec),
                }
            )

        counter += 1
        print(f"Completed {counter} files.")

    minimal_df = (
        pd.DataFrame(rows).sort_values(["time", "plant_id"]).reset_index(drop=True)
    )
    minimal_df.to_csv(output_path, index=False)
    print(f"Saved {len(minimal_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
