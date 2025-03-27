#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 5 -e experiments/offline/O1/E7-DQN-2week-sweeps/PlantSimulator/DQN.json