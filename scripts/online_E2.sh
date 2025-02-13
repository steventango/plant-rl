#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E2/P0/GAC.json -i 0
python src/main_real.py -e experiments/online/E2/P0/Random.json -i 0
