#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 3 -e experiments/offline/O1/E4-linearDQN/linearDQN.json
python3 scripts/local.py --runs 3 -e experiments/offline/O1/E4-linearDQN/linearDQN_sgd.json
