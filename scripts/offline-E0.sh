#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/offline/E0/P5/DQN-1Relu.json 
python scripts/local.py --runs 5 -e experiments/offline/E0/P5/DQN-2Relu.json 