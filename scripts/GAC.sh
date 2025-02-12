#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/offline/GAC/GAC_sweep.json # Remember to change to 5 runs for a sweep