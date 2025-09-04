#!/bin/sh

# exit script on error
set -e

python scripts/local_sim.py --runs 1 --cpus 1 -e experiments/offline/A0/ConstantAgent_MotionTracking.json