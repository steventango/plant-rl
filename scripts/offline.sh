#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/offline/E1/P1/SoftmaxAC.json 