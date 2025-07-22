#!/bin/sh

yes | ffmpeg -framerate 24 -pattern_type glob -i  '/data/online/E10/P1/Constant10/alliance-zone10/processed/v3.6.1/images/*_annotated.jpg' -c:v libx264 -pix_fmt yuv420p /data/online/E10/P1/Constant10/alliance-zone10/processed/v3.6.1/timelapse.mp4 &
yes | ffmpeg -framerate 24 -pattern_type glob -i  '/data/online/E10/P1/Poisson1/alliance-zone01/processed/v3.6.1/images/*_annotated.jpg' -c:v libx264 -pix_fmt yuv420p /data/online/E10/P1/Poisson1/alliance-zone01/processed/v3.6.1/timelapse.mp4 &
wait

# maybe need density area to help filter out bad masks
