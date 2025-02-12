#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E1/P0/Random.json -i 0
python src/main_real.py -e experiments/online/E1/P1/Random.json -i 0
python src/main_real.py -e experiments/online/E1/P2/Random.json -i 0
