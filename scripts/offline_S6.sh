#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P0/OfflineESARSA.json -i 0
 python src/main_real.py -e experiments/offline/S6/P1/ESARSA.json -i 0
