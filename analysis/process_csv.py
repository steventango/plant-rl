import argparse
import ast
import glob
import os
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

FOLDER_PATHS = [
    '/data/plant-rl/online/E16/P1/',
    '/data/plant-rl/online/E16/P2',
]

ACTION_COLS = [f'action.{i}' for i in range(6)]

# needed for plant-rl E16 data only
_AGENT_ACTION_MAP = {
    (1, 0, 0): (1,  11.841679),
    (0, 0, 1): (-1, 59.5),
}

IQR_MIN = 20
IQR_MAX = 80

def iqr_mean(x):
    low, high = np.percentile(x, IQR_MIN), np.percentile(x, IQR_MAX)
    trimmed = x[(x >= low) & (x <= high)]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else 0.0


def decode_agent_action(val, action_0, tol=1.0):
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # handle numpy array repr like "[0 1 0]" (space-separated, no commas)
            parsed = [int(x) for x in val.strip('[] ').split()]
    else:
        parsed = val
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return parsed
    arr = tuple(int(v) for v in parsed)
    if arr not in _AGENT_ACTION_MAP:
        return None
    scalar, expected_action_0 = _AGENT_ACTION_MAP[arr]
    if not np.isclose(action_0, expected_action_0, atol=tol):
        return None
    return scalar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zone', type=int, required=True, help='Zone ID (e.g. 1)')
    args = parser.parse_args()

    zone_id = args.zone
    zone_str = f"alliance-zone{zone_id:02d}"
    output_path = ROOT / f'analysis/data/E16_zone{zone_id:02d}.csv'

    csv_files = []
    for fp in FOLDER_PATHS:
        pattern = os.path.join(fp, '*', zone_str, 'raw_*.csv')
        csv_files.extend(sorted(glob.glob(pattern)))

    print(f"Found {len(csv_files)} CSV files for zone {zone_id}:")
    for f in csv_files:
        print(' ', f)

    rows = []
    counter = 0
    for fpath in csv_files:
        df = pd.read_csv(fpath)

        tz = df['timezone'].dropna().iloc[0]
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df['local_time'] = df['time'].dt.tz_convert(tz)

        mask = df['plant_id'].notna() & (df['local_time'].dt.minute % 10 == 0)
        df = df[mask]

        for ts, group in df.groupby('local_time'):
            group = group[group['area'] > 0]
            row = {
                'time': ts,
                'mean_area': iqr_mean(group['area']),
                'mean_solidity': iqr_mean(group['solidity']),
            }
            for col in ACTION_COLS:
                row[col] = group[col].iloc[0]
            row['agent_action'] = decode_agent_action(group['agent_action'].iloc[0], row['action.0'])
            rows.append(row)
        
        counter += 1
        print(f'Completed {counter} files.')

    minimal_df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    minimal_df.to_csv(output_path, index=False)
    print(f"Saved {len(minimal_df)} rows to {output_path}")


if __name__ == '__main__':
    main()
