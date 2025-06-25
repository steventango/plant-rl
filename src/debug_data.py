import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, Zone
from utils.metrics import UnbiasedExponentialMovingWelford as UEMW


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


def clean_area(df):
    df["clean_area"] = df["area"].copy()
    df["mean"] = 0.0
    df["var"] = 0.0
    for _plant_id, group in df.groupby("plant_id"):
        uemw = UEMW()
        for i, row in group.iterrows():
            # if the area is greater than 2 standard deviations from the moving average
            # repeat the previous value
            mean, var = uemw.compute()
            df.at[i, "mean"] = mean
            df.at[i, "var"] = var
            std = var**0.5
            if row["area"] > mean + 2 * std or row["area"] < mean - 2 * std:
                df.at[i, "clean_area"] = mean
            else:
                # calculate the moving average of area
                uemw.update(row["area"])
    return df["clean_area"]


def main():
    datasets = Path("/data/plant-rl/online/E6/P5").glob("Spreadsheet-Poisson*/z6")
    pipeline_version = "v3.3.0"
    for dataset in datasets:
        csv_path = dataset / "processed" / pipeline_version / "all.csv"
        with open(next(dataset.rglob("config.json"))) as f:
            config = json.load(f)
            zone = get_zone_from_config(config)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])

        df["clean_area"] = clean_area(df)

        for area_col in ["area", "clean_area"]:
            plot_path = (
                dataset / "processed" / pipeline_version / "plots" / f"{area_col}.pdf"
            )
            plot_path.parent.mkdir(exist_ok=True, parents=True)
            # aggregate the data over frame, take average of area
            df_agg = (
                df.groupby(["frame"])
                .agg(
                    {
                        area_col: "mean",
                        "image_name": "first",
                    }
                )
                .reset_index()
            )
            df_agg["area_diff"] = df_agg[area_col].diff()
            df_agg["percent_diff"] = (
                df_agg["area_diff"] / df_agg[area_col].shift(1) * 100
            )

            # Plot the data
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
            ax.plot(df_agg["frame"], df_agg[area_col], label=area_col, color="tab:blue")
            ax.set_title(f"Plant Area Over Time ({dataset.name})")
            ax.set_xlabel("Frame")
            ax.set_ylabel(area_col)
            ax.legend()
            ax.tick_params(axis="x", rotation=45)
            sns.despine(ax=ax)

            # Second subplot: plot percent_diff
            ax2.plot(
                df_agg["frame"],
                df_agg["percent_diff"],
                label="percent_diff",
                color="tab:orange",
            )
            ax2.set_ylabel(f"{area_col} Diff")
            ax2.set_title("Area Diff Over Time")
            ax2.legend()
            ax2.set_xlabel("Frame")
            ax2.tick_params(axis="x", rotation=45)
            sns.despine(ax=ax2)

            plt.tight_layout()
            plt.savefig(plot_path)
            print(plot_path)

            # Print out the image_name for rows with a big jump in area
            path_prefix = dataset / "processed" / pipeline_version / "images"
            # sort descending by percent_diff
            df_agg = df_agg.sort_values(by="percent_diff", ascending=False)
            for _i, row in df_agg.head(10).iterrows():
                _, prev_row = next(
                    df_agg[df_agg["frame"] == row["frame"] - 1].iterrows()
                )
                print(f"Jump detected at frame {row['frame']}!")
                print(f"Area current: {row[area_col]}")
                print(f"Area previous: {prev_row[area_col]}")
                print(f"Percent diff: {row['percent_diff']}")
                out_dir = dataset / "processed" / pipeline_version
                out_dir_images = out_dir / "images"
                out_dir_images.mkdir(exist_ok=True, parents=True)
                image_path = dataset / "images" / row["image_name"]
                process_one_image(zone, out_dir_images, image_path, 0)
                prev_image_path = dataset / "images" / prev_row["image_name"]
                process_one_image(zone, out_dir_images, prev_image_path, 0)

                prev_image_path = (
                    path_prefix
                    / f"{prev_row['image_name'].replace('_right.jpg', '_shape_image.jpg')}"
                )
                image_path = (
                    path_prefix
                    / f"{row['image_name'].replace('_right.jpg', '_shape_image.jpg')}"
                )
                prev_image_path2 = (
                    path_prefix
                    / f"{prev_row['image_name'].replace('_right.jpg', 'masks2.jpg')}"
                )
                image_path2 = (
                    path_prefix
                    / f"{row['image_name'].replace('_right.jpg', 'masks2.jpg')}"
                )
                prev_image_path3 = (
                    path_prefix
                    / f"{prev_row['image_name'].replace('_right.jpg', 'boxes.jpg')}"
                )
                image_path3 = (
                    path_prefix
                    / f"{row['image_name'].replace('_right.jpg', 'boxes.jpg')}"
                )

                img1 = Image.open(prev_image_path)
                img2 = Image.open(image_path)
                img3 = Image.open(prev_image_path2)
                img4 = Image.open(image_path2)
                img5 = Image.open(prev_image_path3)
                img6 = Image.open(image_path3)
                img = Image.new("RGB", (img1.width + img2.width, 3 * img1.height))
                img.paste(img1, (0, 0))
                img.paste(img2, (img1.width, 0))
                img.paste(img3, (0, img1.height))
                img.paste(img4, (img1.width, img1.height))
                img.paste(img5, (0, 2 * img1.height))
                img.paste(img6, (img1.width, 2 * img1.height))

                # draw a line between the two images
                draw = ImageDraw.Draw(img)
                draw.line(
                    (img1.width, 0, img1.width, img1.height), fill="black", width=5
                )
                # draw horizontal lines between the two images
                draw.line(
                    (0, img1.height, img.width, img1.height), fill="black", width=5
                )
                draw.line(
                    (0, 2 * img1.height, img.width, 2 * img1.height),
                    fill="black",
                    width=5,
                )

                # save to debug folder
                debug_path = (
                    dataset / "processed" / pipeline_version / "debug" / area_col
                )
                debug_path.mkdir(exist_ok=True, parents=True)
                debug_image_path = debug_path / f"jump_{row['frame']}.jpg"
                img.save(debug_path / f"jump_{row['frame']}.jpg")
                print(f"Saved debug image to {debug_image_path}")


def process_one_image(zone, out_dir, path, index):
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
