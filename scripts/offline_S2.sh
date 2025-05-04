#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/offline/S2/P1/ESARSA.json -i 0
