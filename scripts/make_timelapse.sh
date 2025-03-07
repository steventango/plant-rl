#!/bin/sh

ffmpeg -framerate 24 -pattern_type glob -i 'data/first_exp/z2cR/*.png' -c:v libx264 -pix_fmt yuv420p timelapse.mp4
