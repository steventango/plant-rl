#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/offline/E2/P1/Constant.json
python scripts/local.py --runs 5 -e experiments/offline/E2/P1/Random.json
python scripts/local.py --runs 5 -e experiments/offline/E2/P1/GAC-sweep.json --cpus 1
python experiments/offline/E2/compare-alg.py
