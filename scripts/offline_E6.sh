#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/offline/E6/P0/tc-ESARSA.json -i 0
