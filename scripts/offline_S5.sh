#!/bin/sh

# exit script on error
set -e


python scripts/local_sim.py --runs 5 --cpus 30 -e experiments/offline/S5/P2/ReplayESARSA.json
