#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S7/P0/BatchESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S7/P1/BatchESARSA.json -i 0 &
wait
