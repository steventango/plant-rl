#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 --cpus 30 -e experiments/offline/S1/P0/tc-ESARSA.json
python scripts/local.py --runs 1 --cpus 1 -e experiments/offline/S1/P0/optimal-sequence.json
