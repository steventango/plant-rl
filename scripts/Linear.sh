#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 5 -e experiments/offline/O1/E8-trivial_rew_1ep/DQN.json
python3 scripts/local.py --runs 5 -e experiments/offline/O1/E8-trivial_rew_1ep/GAC.json

