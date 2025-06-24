import glob
import os
import re
from datetime import datetime

import cv2
import numpy as np
from imageio.v2 import imread
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm.contrib.concurrent import process_map

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import load_zone_from_config


def get_key(value):
    return int(re.findall(r"\d+", value)[-1])


def get_image(image_path, zone_identifier: str):
    image = imread(image_path)
    iso_format = os.path.basename(image_path).split("_")[0]
    timestamp = datetime.fromisoformat(iso_format)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    debug_images = {}
    zone = load_zone_from_config(zone_identifier)
    process_image(image, zone.trays, debug_images)
    shape_image = debug_images["shape_image"]
    shape_image = np.array(shape_image)
    text_size = cv2.getTextSize(timestamp_str, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)[0]
    text_x = (shape_image.shape[1] - text_size[0]) // 2
    text_y = 50
    shape_image = cv2.putText(
        shape_image,
        timestamp_str,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return shape_image


def make_timelapse(pattern, output_video, zone):
    image_files = sorted(glob.glob(pattern), key=get_key)
    images = process_map(get_image, image_files, [zone] * len(image_files), max_workers=8)
    clip = ImageSequenceClip(images, fps=1)
    clip.write_videofile(output_video)


def main():
    for zone in ["mitacs-zone01", "mitacs-zone02", "mitacs-zone06", "mitacs-zone09"]:
        zone_number = int(zone.split("-")[-1].replace("zone", ""))
        pattern = f"data/online/E6/P3/Spreadsheet-{zone_number}/z{zone_number}/*.jpg"
        make_timelapse(pattern, output_video=f"timelapse_{zone}.mp4", zone=zone)


if __name__ == "__main__":
    main()
