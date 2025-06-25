from itertools import chain
from pathlib import Path

import pandas as pd

# nazmus_exp/<dataset>/
dataset_paths = sorted(
    chain(
        Path("/data/maria_exp/").glob("*/"),
    )
)

print("Dataset paths:")
for dataset_path in dataset_paths:
    print(dataset_path)

for dataset_path in dataset_paths:
    # Get the name of the dataset
    dataset_name = dataset_path.stem
    # Get the path to the images directory
    images_path = dataset_path / "images"

    # Get the image names from the images directory
    image_names = sorted(images_path.glob("*.jpg"))

    # Create a new DataFrame with the image names
    df = pd.DataFrame({"image_name": [image_name.name for image_name in image_names]})
    # frame,time,image_name
    df["frame"] = df.index
    # nazmus_exp/z11c2/images/z11c2--2022-07-22--00-00-01.jpg
    df["time"] = [image_name.stem.split("-", 1)[1] for image_name in image_names]
    # convert time to datetime
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d-%H-%M-%S")
    # convert from America/Edmonton to UTC
    df["time"] = (
        df["time"]
        .dt.tz_localize("America/Edmonton", ambiguous=True)
        .dt.tz_convert("UTC")
    )

    # Save the new DataFrame to a CSV file
    df.to_csv(dataset_path / "raw.csv", index=False)
