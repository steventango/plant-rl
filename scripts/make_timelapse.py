import argparse
import os
import glob
import re

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from imageio.v2 import imread
from tqdm.contrib.concurrent import thread_map


def get_key(value):
    return int(re.findall(r'\d+', value)[-1])


def make_timelapse(pattern, output_video):
    image_files = sorted(glob.glob(pattern), key=get_key)
    images = thread_map(imread, image_files, desc="Loading images")
    clip = ImageSequenceClip(images, fps=24)
    clip.write_videofile(output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a timelapse video from photos.")
    parser.add_argument("pattern", type=str, nargs='?', default="results/online/E4/P0/Spreadsheet/0/images/*.png", help="The glob pattern to match video files.")
    parser.add_argument("output_video", type=str, nargs='?', default="timelapse.mp4", help="The name of the output video file.")
    args = parser.parse_args()

    make_timelapse(args.pattern, args.output_video)
