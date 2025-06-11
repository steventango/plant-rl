#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P1/OfflineESARSA.json -i 0
