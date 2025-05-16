#!/bin/sh

# exit script on error
set -e


python scripts/local_sim.py --runs 5 --cpus 5 -e experiments/offline/S5/P0/ESARSA_replay.json &
python scripts/local_sim.py --runs 5 --cpus 5 -e experiments/offline/S5/P0/ESARSA_replay_no_decay.json &
python scripts/local_sim.py --runs 5 --cpus 5 -e experiments/offline/S5/P0/ESARSA_replay_no_decay_big_batch.json &
python scripts/local_sim.py --runs 5 --cpus 5 -e experiments/offline/S5/P0/ESARSA_replay_ratio_16.json &
wait
