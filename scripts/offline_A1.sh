#!/bin/sh

# exit script on error
set -e

#python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/A1/RandomAgent_GPsim.json
python scripts/local.py --runs 1 --cpus 1 -e experiments/offline/A1/InAC_GPsim0.json