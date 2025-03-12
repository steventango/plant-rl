#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 1 -e experiments/offline/linear/linear2/MultiPlantSimulator/constant.json # Remember to change to 5 runs for a sweep
python3 scripts/local.py --runs 5 -e experiments/offline/linear/linear2/MultiPlantSimulator/QL.json # Remember to change to 5 runs for a sweep
#python3 scripts/local.py --runs 5 -e experiments/offline/linear/MultiPlantSimulator/ESARSA.json # Remember to change to 5 runs for a sweep