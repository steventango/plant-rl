#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P1/OfflineESARSA.json -i 0
python src/main_offline.py -e experiments/offline/S6/P2/OfflineESARSA.json -i 0
python src/main_offline.py -e experiments/offline/S6/P3/OfflineESARSA.json -i 0
python src/main_offline.py -e experiments/offline/S6/P4/OfflineESARSA.json -i 0
python src/main_offline.py -e experiments/offline/S6/P5/OfflineESARSA.json -i 0
python src/main_offline.py -e experiments/offline/S6/P6/OfflineESARSA.json -i 0
