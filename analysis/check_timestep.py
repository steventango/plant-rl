import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz


def get_yesterday():
    return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")


def count_timesteps_in_file(filepath):
    df = pd.read_csv(filepath)
    # Convert 'time' column to datetime in UTC, then localize to America/Edmonton
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    # Drop rows with invalid time
    df = df.dropna(subset=["time"])
    # Convert to America/Edmonton timezone
    df["time_local"] = df["time"].dt.tz_convert("America/Edmonton")
    # July 5, 2025 in America/Edmonton
    # tz = pytz.timezone('America/Edmonton')
    # start = tz.localize(datetime.datetime(2025, 7, 5, 9, 0, 0))
    # end = tz.localize(datetime.datetime(2025, 7, 5, 23, 59, 59))
    # mask = (df['time_local'] >= start) & (df['time_local'] < end)
    # filtered = df[mask]
    # Get unique timesteps based on the 'time' column only
    unique_timesteps = np.sort(np.array(df["time_local"].unique()))
    return unique_timesteps


def main():
    patterns = [
        "/data/online/A0/**/raw.csv",
        "/data/online/E9/**/raw.csv",
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
        if not files:
            print(f"No raw.csv files found with pattern: {pattern}")
            return

    from collections import defaultdict

    zone_timesteps = defaultdict(list)
    tz = pytz.timezone("America/Edmonton")
    start = tz.localize(datetime.datetime(2025, 7, 4, 0, 0, 0))
    end = tz.localize(datetime.datetime(2025, 7, 6, 23, 59, 59))
    expected_times = pd.date_range(start=start, end=end, freq="min")
    import re

    zone_pattern = re.compile(r"zone(\d+)", re.IGNORECASE)
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        # Extract zone from filename
        match = zone_pattern.search(filepath)
        if match:
            zone = f"zone{match.group(1)}"
        else:
            zone = "unknown"
        unique_timesteps = count_timesteps_in_file(filepath)
        zone_timesteps[zone].extend(unique_timesteps)
        print(f"{filepath} (zone: {zone}): {len(unique_timesteps)} filtered timesteps")

    # Aggregate and deduplicate timesteps per zone
    for zone in zone_timesteps:
        # Floor to minute and deduplicate
        zone_timesteps[zone] = np.unique(
            pd.to_datetime(zone_timesteps[zone]).floor("min")  # type: ignore
        ).tolist()

    # Plotting
    zones = sorted(zone_timesteps.keys())
    n = len(zones)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, zone in enumerate(zones):
        timesteps = zone_timesteps[zone]
        present = pd.Series(0, index=expected_times)
        present.loc[present.index.isin(timesteps)] = 1
        axes[i].imshow(
            [present.values],
            aspect="auto",
            cmap="Greys",
            extent=[0, len(expected_times), 0, 1],
        )
        axes[i].set_yticks([])
        axes[i].set_ylabel(zone)
        axes[i].set_xlim(0, len(expected_times))
        axes[i].set_title(f"Missing timesteps for {zone}")
    axes[-1].set_xticks(np.linspace(0, len(expected_times), 10))
    axes[-1].set_xticklabels(
        [t.strftime("%H:%M") for t in expected_times[:: len(expected_times) // 10]]
    )
    plt.xlabel("Time (America/Edmonton)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
