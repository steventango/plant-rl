#!/bin/sh

# exit script on error
set -e

python src/main_offline.py -e experiments/offline/S6/P17/OfflineESARSA.json -i 0
