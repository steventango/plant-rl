import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import deserialize_zone


def main():
    datasets = [
        "/data/online/E4/P0.2/z2",
        "/data/online/E4/P1/z2"
    ]
    datasets = [Path(dataset) for dataset in datasets]
    datasets = sorted(datasets)

    pipeline_version = "v3.6.0"
    for dataset in datasets:
        with open(dataset / "config.json") as f:
            config = json.load(f)
            zone = deserialize_zone(config["zone"])
        core_df = pd.read_csv(dataset / "core.csv")
        out_dir = dataset / "processed" / pipeline_version
        out_dir_images = out_dir / "images"
        out_dir_images.mkdir(exist_ok=True, parents=True)
        images_dir = dataset / "images"
        image_paths = images_dir / core_df["image_name"]

        # Create list of arguments for parallel processing
        process_args = [(zone, out_dir_images, path, i) for i, path in enumerate(image_paths)]

        # Use process_map to parallelize the processing
        results = process_map(
            process_one_image,
            process_args,
            max_workers=1,
            chunksize=1,
        )

        # Collect results
        new_df = pd.concat(results)
        plant_ids = new_df["plant_id"].unique()
        # add plant_id column to core_df, repeat rows for each plant_id
        core_df = core_df.loc[core_df.index.repeat(len(plant_ids))]
        # add plant_id column to core_df
        core_df["plant_id"] = np.tile(plant_ids, len(core_df) // len(plant_ids))

        df = pd.merge(
            core_df,
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
    for key, image in debug_images.items():
        image.save(out_dir / f"{isoformat}_{key}.jpg")
    return df


if __name__ == "__main__":
    main()
