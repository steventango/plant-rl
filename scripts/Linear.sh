#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 5 -e experiments/offline/O1/E8-ESARSA-8wk/ESARSA.json
python3 scripts/local.py --runs 5 -e experiments/offline/O1/E8-ESARSA-4wk-lr_sweep/ESARSA.json


