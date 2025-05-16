#!/bin/sh

# exit script on error
set -e

python scripts/local_sim.py --runs 5 --cpus 1 -e experiments/offline/E5/P0/ESARSA_lambda.json