#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P15/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P16/OfflineESARSA.json -i 0 &
wait
