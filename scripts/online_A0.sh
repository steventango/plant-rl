#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/A0/P0/MotionTrackingController.json -i 0
