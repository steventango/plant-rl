#!/bin/sh

# exit script on error
set -e
python scripts/local.py --runs 3 --cpus 1 --entry src/main_real.py -e experiments/online/E2/P0/GAC.json
python scripts/local.py --runs 3 --cpus 1 --entry src/main_real.py -e experiments/online/E2/P0/Random.json
python scripts/local.py --runs 3 --cpus 1 --entry src/main_real.py -e experiments/online/E2/P0/Constant.json
