#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 5 -e experiments/offline/linear/MultiPlantSimulator/QL.json # Remember to change to 5 runs for a sweep