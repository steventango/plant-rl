import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, Zone


def get_zone_from_config(config):
    config_zone = config["zone"]
    zone = Zone(
        identifier=config_zone["identifier"],
        camera_left_url=config_zone["camera_left_url"],
        camera_right_url=config_zone["camera_right_url"],
        lightbar_url=config_zone["lightbar_url"],
        trays=[
            Tray(
                n_wide=tray["n_wide"],
                n_tall=tray["n_tall"],
                rect=Rect(
                    top_left=tuple(tray["rect"]["top_left"]),
                    top_right=tuple(tray["rect"]["top_right"]),
                    bottom_left=tuple(tray["rect"]["bottom_left"]),
                    bottom_right=tuple(tray["rect"]["bottom_right"]),
                ),
            )
            for tray in config_zone["trays"]
        ],
    )
    return zone


def main():
    datasets = Path("/data/plant-rl/online/E6/P5").glob("Spreadsheet-Poisson*/z*")
    datasets = sorted(datasets)
    datasets = datasets[2:]
    pipeline_version = "v3.3.1"
    for dataset in datasets:
        with open(next(dataset.rglob("config.json"))) as f:
            config = json.load(f)
            zone = get_zone_from_config(config)
        raw_df = pd.read_csv(dataset / "raw.csv")
        out_dir = dataset / "processed" / pipeline_version
        out_dir_images = out_dir / "images"
        out_dir_images.mkdir(exist_ok=True, parents=True)
        paths = sorted((dataset / "images").glob("*.jpg"))[1:]

        dfs = []
        for i, path in tqdm(enumerate(paths), total=len(paths)):
            df = process_one_image(zone, out_dir_images, path, i)
            dfs.append(df)

        new_df = pd.concat(dfs)
        # raw_df drop cols that are in new_df
        cols_to_drop = [col for col in raw_df.columns if col in new_df.columns]
        cols_to_drop.remove("frame")
        cols_to_drop.remove("plant_id")
        raw_df = raw_df.drop(columns=cols_to_drop)
        # TODO: temporary
        # if min plant_id is 0, add 1 to all plant_ids
        if  raw_df["plant_id"].min() == 0:
            raw_df["plant_id"] += 1
        df = pd.merge(
            raw_df,
            new_df,
            how="left",
            left_on=["frame", "plant_id"],
            right_on=["frame", "plant_id"],
        )
        df.to_csv(out_dir / "all.csv", index=False)


def process_one_image(zone, out_dir, path, index):
    isoformat = path.stem.split("_")[0]
    image = np.array(Image.open(path))
    debug_images = {}
    df = process_image(image, zone.trays, debug_images)
    df["frame"] = index
    for key, image in debug_images.items():
        image.save(out_dir / f"{isoformat}_{key}.jpg")
    return df


if __name__ == "__main__":
    main()
