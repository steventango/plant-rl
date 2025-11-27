#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/offline/A1/P0/InAC_GPsim.json