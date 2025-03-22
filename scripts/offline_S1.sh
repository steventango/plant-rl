#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 --cpus 5 -e experiments/offline/S1/P1/optimal-sequence.json
python scripts/local.py --runs 5 --cpus 30 -e experiments/offline/S1/P1/tc-ESARSA.json
