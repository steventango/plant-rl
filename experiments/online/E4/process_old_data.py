import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, Zone
from utils.metrics import iqm

timeframes = [
    {"date": "2025-02-23", "ontime": "09:01", "offtime": "21:01"},
    {"date": "2025-02-24", "ontime": "09:01", "offtime": "21:01"},
    {"date": "2025-02-25", "ontime": "09:01", "offtime": "21:01"},
    {"date": "2025-02-26", "ontime": "09:01", "offtime": "21:01"},
    {"date": "2025-02-27", "ontime": "09:01", "offtime": "21:01"},
    {"date": "2025-02-28", "ontime": "09:01", "offtime": "21:01"},
    {"date": "2025-03-01", "ontime": "13:01", "offtime": "23:59"},
    {"date": "2025-03-02", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-02", "ontime": "13:01", "offtime": "23:59"},
    {"date": "2025-03-03", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-03", "ontime": "13:01", "offtime": "23:59"},
    {"date": "2025-03-04", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-04", "ontime": "13:01", "offtime": "16:01"},
    {"date": "2025-03-04", "ontime": "18:01", "offtime": "23:59"},
    {"date": "2025-03-05", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-05", "ontime": "13:01", "offtime": "16:01"},
    {"date": "2025-03-05", "ontime": "18:01", "offtime": "23:59"},
    {"date": "2025-03-06", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-06", "ontime": "13:01", "offtime": "16:01"},
    {"date": "2025-03-06", "ontime": "18:01", "offtime": "23:59"},
    {"date": "2025-03-07", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-07", "ontime": "13:01", "offtime": "16:01"},
    {"date": "2025-03-07", "ontime": "18:01", "offtime": "23:59"},
    {"date": "2025-03-08", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-08", "ontime": "13:01", "offtime": "16:01"},
    {"date": "2025-03-08", "ontime": "18:01", "offtime": "23:59"},
    {"date": "2025-03-09", "ontime": "00:00", "offtime": "01:01"},
    {"date": "2025-03-09", "ontime": "14:01", "offtime": "17:01"},
    {"date": "2025-03-09", "ontime": "19:01", "offtime": "23:59"},
    {"date": "2025-03-10", "ontime": "00:00", "offtime": "02:01"},
    {"date": "2025-03-10", "ontime": "14:01", "offtime": "23:59"},
    {"date": "2025-03-11", "ontime": "00:00", "offtime": "02:01"},
]


def main():
    dfs = []
    zone = Zone(
        identifier=2,
        camera_left_url=None,
        camera_right_url="http://mitacs-zone02-camera02.ccis.ualberta.ca:8080/observation",
        lightbar_url="http://mitacs-zone2.ccis.ualberta.ca:8080/action",
        trays=[
            Tray(
                n_wide=4,
                n_tall=4,
                rect=Rect(
                    top_left=(1241, 978),
                    top_right=(2017, 952),
                    bottom_left=(1258, 1804),
                    bottom_right=(1972, 1667),
                ),
            )
        ],
    )
    zone_dir = Path("data/first_exp/z2cR")
    out_dir = Path("results") / zone_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    paths = sorted(zone_dir.glob("*.png"))
    paths = [
        path
        for path in paths
        if datetime.datetime.fromisoformat(path.stem).minute % 5 == 0
    ]
    paths = [
        path
        for path in paths
        if in_timeframe(datetime.datetime.fromisoformat(path.stem))
    ]

    # Use process_map to parallelize the processing
    results = process_map(
        process_one_image,
        [(zone, out_dir, path) for path in paths],
        max_workers=12,
        chunksize=10,
    )

    # Collect results
    for df in results:
        if df is not None:
            dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(out_dir / "data.csv", index=False)

    df = df.pivot(index="timestamp", columns="plant_id", values="area")
    df.to_csv(out_dir / "area_over_time.csv")


def in_timeframe(timestamp):
    for timeframe in timeframes:
        ontime = datetime.datetime.fromisoformat(
            f"{timeframe['date']}T{timeframe['ontime']}"
        )
        offtime = datetime.datetime.fromisoformat(
            f"{timeframe['date']}T{timeframe['offtime']}"
        )
        if ontime <= timestamp <= offtime:
            return True
    return False


def process_one_image(args):
    zone, out_dir, path = args
    timestamp = datetime.datetime.fromisoformat(path.stem)
    image = np.array(Image.open(path))
    debug_images = {}
    df, _ = process_image(image, zone.trays, debug_images)
    df["timestamp"] = timestamp

    avg = iqm(jnp.array(df["area"]), 0.05)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {"plant_id": ["iqm"], "area": [avg], "timestamp": [timestamp]}
            ),
        ]
    )
    df.to_csv(out_dir / f"{path.stem}.csv", index=False)
    for key, value in debug_images.items():
        value.save(out_dir / f"{path.stem}_{key}.png")
    return df


if __name__ == "__main__":
    main()
