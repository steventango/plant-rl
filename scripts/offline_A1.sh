#!/bin/sh

# exit script on error
set -e

python scripts/slurm.py --cluster clusters/compute_gpu.json --runs 5 -e experiments/offline/A1/P0/InAC_GPsim.json