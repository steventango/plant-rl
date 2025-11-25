#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 1 --cpus 1 -e experiments/offline/A2/InAC_GPsim0.json