#!/bin/sh

# exit script on error
set -e

python scripts/local_sim.py --runs 5 --cpus 5 -e experiments/offline/S2/P2/ESARSA0_TOD.json &
python scripts/local_sim.py --runs 5 --cpus 5 -e experiments/offline/S2/P2/ESARSABoltzmann.json &
python scripts/local_sim.py --runs 1 --cpus 5 -e experiments/offline/S2/P2/Context_Bandit.json &
wait
