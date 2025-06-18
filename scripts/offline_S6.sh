#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P8/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P9/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P10/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P11/OfflineESARSA.json -i 0 &
wait
