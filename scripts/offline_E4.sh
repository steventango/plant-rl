#!/bin/sh

# exit script on error
set -e

python scripts/local_sim.py --runs 5 --cpus 1 -e experiments/offline/E4/P8/Bandit.json
python scripts/local_sim.py --runs 5 --cpus 1 -e experiments/offline/E4/P8/Context_Bandit.json
python scripts/local_sim.py --runs 5 --cpus 1 -e experiments/offline/E4/P8/ESARSA0_TOD.json
