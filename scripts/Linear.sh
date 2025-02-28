#!/bin/sh

# exit script on error
set -e

python3 scripts/local.py --runs 1 -e experiments/offline/linear/CliffWalking/QL.json --cpus 1 # Remember to change to 5 runs for a sweep