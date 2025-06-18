#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P11/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P12/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P13/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P14/OfflineESARSA.json -i 0 &
wait
