#!/bin/sh

# exit script on error
set -e
python scripts/local.py --runs 1 --cpus 1 --entry src/main_real.py -e experiments/online/E3/P0/GAC.json

python experiments/online/E2/P0/compare-alg.py
