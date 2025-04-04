#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/offline/E6/P0/tc-ESARSA.json -i 0
python scripts/local.py --runs 30 --cpus 20 -e experiments/offline/E6/P1/ESARSA.json
python src/main_real.py -e experiments/offline/E6/P2/tc-ESARSA.json -i 0
