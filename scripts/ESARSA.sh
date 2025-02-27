#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 1 -e experiments/offline/linear/CliffWalking/ESARSA.json --cpus 1 # Remember to change to 5 runs for a sweep