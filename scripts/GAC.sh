#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 20 -e experiments/offline/GAC/acrobot/GAC.json --cpus 1 # Remember to change to 5 runs for a sweep