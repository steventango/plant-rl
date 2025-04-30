#!/bin/sh

ffmpeg -framerate 24 -pattern_type glob -i  '/data/nazmus_exp/z11c1/processed/v3.6.0/images/*_stacked.jpg' -c:v libx264 -pix_fmt yuv420p timelapse_nazmusz11c1.mp4

# maybe need density area to help filter out bad masks
