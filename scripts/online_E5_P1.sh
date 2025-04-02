#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E5/P1/Poisson.json -i 1
python src/main_real.py -e experiments/online/E5/P1/Poisson-slow.json -i 1
