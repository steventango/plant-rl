#!/bin/sh

ffmpeg -framerate 24 -pattern_type glob -i '/data/plant-rl/online/E6/P5/Spreadsheet-Poisson6/z6/processed/v3.3.1/images/*_stacked.jpg' -c:v libx264 -pix_fmt yuv420p timelapse_6.mp4
ffmpeg -framerate 24 -pattern_type glob -i '/data/plant-rl/online/E6/P5/Spreadsheet-Poisson9/z9/processed/v3.3.1/images/*_stacked.jpg' -c:v libx264 -pix_fmt yuv420p timelapse_9.mp4
