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

# needed for plant-rl E17 data only
_AGENT_ACTION_MAP = [
    (-1, [59.5, 38.06567251461989, 4.161520467836257, 0.0, 3.2728070175438595, 0.0]),
    (1, [11.841678939617085, 43.43770741286205, 4.748816887579775, 0.0, 46.15, 0.0]),
]

ACTION_TOL = 0.01

IQR_MIN = 0  # the problematic area readings are null
IQR_MAX = 100


def iqr_mean(x):
    low, high = np.percentile(x, IQR_MIN), np.percentile(x, IQR_MAX)
    trimmed = x[(x >= low) & (x <= high)]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else 0.0


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
    output_path = ROOT / f"analysis/data/E17_zone{zone_id:02d}.csv"

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

        mask = df["plant_id"].notna() & (df["local_time"].dt.minute % 10 == 0)
        df = df[mask]

        for ts, group in df.groupby("local_time"):
            group = group[group["area"] > 0]
            row = {
                "time": ts,
                "mean_area": iqr_mean(group["area"]),
                "mean_solidity": iqr_mean(group["solidity"]),
            }
            for col in ACTION_COLS:
                row[col] = group[col].iloc[0]
            row["agent_action"] = decode_agent_action([row[col] for col in ACTION_COLS])
            rows.append(row)

        counter += 1
        print(f"Completed {counter} files.")

    minimal_df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    minimal_df.to_csv(output_path, index=False)
    print(f"Saved {len(minimal_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
