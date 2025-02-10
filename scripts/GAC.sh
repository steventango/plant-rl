#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 1 -e experiments/offline/GAC/GAC.json