#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P19/OfflineESARSA.json -i 0 &
python src/main_offline.py -e experiments/offline/S6/P20/OfflineESARSA.json -i 0 &
wait
