import datetime
from itertools import chain
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, Zone


def main():
    method = "grounded-sam2"
    dfs = []
    zone = Zone(
            identifier=1,
            camera_left_url="http://mitacs-zone01-camera02.ccis.ualberta.ca:8080/observation",
            camera_right_url=None,
            lightbar_url="http://mitacs-zone1.ccis.ualberta.ca:8080/action",
            trays=[
                Tray(
                    n_wide=6,
                    n_tall=4,
                    rect=Rect(
                        top_left=(612, 44),
                        top_right=(1882, 24),
                        bottom_left=(560, 888),
                        bottom_right=(1918, 908),
                    ),
                ),
                Tray(
                    n_wide=6,
                    n_tall=4,
                    rect=Rect(
                        top_left=(552, 974),
                        top_right=(1892, 1002),
                        bottom_left=(604, 1742),
                        bottom_right=(1814, 1800),
                    ),
                ),
            ],
        )
    zone_poisson = Zone(
        identifier=1,
        camera_left_url="http://mitacs-zone01-camera02.ccis.ualberta.ca:8080/observation",
        camera_right_url=None,
        lightbar_url="http://mitacs-zone1.ccis.ualberta.ca:8080/action",
        trays=[
            Tray(
                n_wide=6,
                n_tall=3,
                rect=Rect(
                    top_left=(528, 232),
                    top_right=(1806, 195),
                    bottom_left=(504, 843),
                    bottom_right=(1815, 882),
                ),
            ),
            Tray(
                n_wide=6,
                n_tall=3,
                rect=Rect(
                    top_left=(489, 927),
                    top_right=(1791, 978),
                    bottom_left=(513, 1512),
                    bottom_right=(1731, 1626),
                ),
            ),
        ],
    )
    zone_dirs = [
        Path("/data/plant-rl/online/E5/P0/Spreadsheet/z1"),
        # Path("/data/plant-rl/online/E5/P1/Spreadsheet-Poisson/z1")
    ]
    out_dir = Path(f"tmp/{method}")
    out_dir.mkdir(exist_ok=True, parents=True)
    paths = []
    for zone_dir in zone_dirs:
        candidate_paths = zone_dir.glob("*.jpg")
        candidate_paths = sorted(candidate_paths)
        timestamp_last = None
        for path in candidate_paths:
            timestamp = datetime.datetime.fromisoformat(path.stem.split("_")[0])
            if timestamp < datetime.datetime(2025, 3, 18, 10, 15, 0):
                continue
            if timestamp_last is None or (timestamp - timestamp_last).total_seconds() >= 5 * 60:
                paths.append(path)
                timestamp_last = timestamp

    for path in tqdm(paths):
        df = process_one_image(zone, out_dir, path)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(out_dir / "all.csv", index=False)


def process_one_image(zone, out_dir, path):
    isoformat = path.stem.split("_")[0]
    timestamp = datetime.datetime.fromisoformat(isoformat)
    image = np.array(Image.open(path))
    debug_images = {}
    df = process_image(image, zone.trays, debug_images)
    for key, image in debug_images.items():
        image.save(out_dir / f"{isoformat}_{key}.jpg")
    df["timestamp"] = timestamp
    return df


if __name__ == "__main__":
    main()
