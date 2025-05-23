#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 --cpus 5 -e experiments/offline/S1/P2/constant.json
python scripts/local.py --runs 5 --cpus 5 -e experiments/offline/S1/P2/optimal-sequence.json
python scripts/local.py --runs 5 --cpus 30 -e experiments/offline/S1/P2/tc-ESARSA.json
