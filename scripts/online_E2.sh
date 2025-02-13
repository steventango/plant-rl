#!/bin/sh

# exit script on error
set -e

for i in 0 1 2; do
  python src/main_real.py -e experiments/online/E2/P0/GAC.json -i $i
  python src/main_real.py -e experiments/online/E2/P0/Random.json -i $i
  python src/main_real.py -e experiments/online/E2/P0/Constant.json -i $i
done
