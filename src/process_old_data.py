import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, Zone


def get_zone_from_config(config):
    config_zone = config["zone"]
    zone = Zone(
        identifier=config_zone["identifier"],
        camera_left_url=config_zone.get("camera_left_url"),
        camera_right_url=config_zone.get("camera_right_url"),
        lightbar_url=config_zone.get("lightbar_url"),
        calibration=None,
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
    datasets = Path("/data").glob("nazmus_exp/z11c1")
    datasets = sorted(datasets)

    pipeline_version = "v3.6.0"
    for dataset in datasets:
        with open(next(dataset.rglob("config.json"))) as f:
            config = json.load(f)
            zone = get_zone_from_config(config)
        try:
            raw_df = pd.read_csv(dataset / "raw.csv")
        except FileNotFoundError:
            print(f"raw.csv not found in {dataset}.")
            raw_df = pd.DataFrame()
        out_dir = dataset / "processed" / pipeline_version
        out_dir_images = out_dir / "images"
        out_dir_images.mkdir(exist_ok=True, parents=True)
        paths = sorted((dataset / "images").glob("*.jpg"))[::2]

        # Create list of arguments for parallel processing
        process_args = [(zone, out_dir_images, path, i) for i, path in enumerate(paths)]

        # Use process_map to parallelize the processing
        results = process_map(
            process_one_image,
            process_args,
            max_workers=4,
            chunksize=1,
        )

        # Collect results
        new_df = pd.concat(results)

        if raw_df.empty:
            # if raw_df is empty, just save new_df
            new_df["time"] = pd.to_datetime(
                new_df["image_name"].str.extract(
                    r"--(\d{4}-\d{2}-\d{2})--(\d{2}-\d{2}-\d{2})"
                )[0]
                + " "
                + new_df["image_name"].str.extract(r"--(\d{2}-\d{2}-\d{2})")[0],
                format="%Y-%m-%d %H-%M-%S",
            )
            new_df.to_csv(out_dir / "all.csv", index=False)
            continue

        # raw_df drop cols that are in new_df
        cols_to_drop = [col for col in raw_df.columns if col in new_df.columns]
        cols_to_drop.remove("frame")
        cols_to_drop.remove("plant_id")
        raw_df = raw_df.drop(columns=cols_to_drop)
        # TODO: temporary
        # if min plant_id is 0, add 1 to all plant_ids
        if raw_df["plant_id"].min() == 0:
            raw_df["plant_id"] += 1
        df = pd.merge(
            raw_df,
            new_df,
            how="left",
            left_on=["frame", "plant_id"],
            right_on=["frame", "plant_id"],
        )
        df.to_csv(out_dir / "all.csv", index=False)


def process_one_image(args):
    zone, out_dir, path, index = args
    isoformat = path.stem.split("_")[0]
    image = np.array(Image.open(path))
    debug_images = {}
    df, _ = process_image(image, zone.trays, debug_images)
    df["frame"] = index
    df["image_name"] = path.name
    for key, image in debug_images.items():
        image.save(out_dir / f"{isoformat}_{key}.jpg")
    return df


if __name__ == "__main__":
    main()
