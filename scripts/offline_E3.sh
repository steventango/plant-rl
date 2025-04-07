#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/E3/P16/tc-ESARSA.json
