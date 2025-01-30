#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/offline/E0/P3/DQN-1Relu.json 