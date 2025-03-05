#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/E2/P13/GAC-sweep-s2.json 
python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/E2/P13/GAC-sweep-s3.json 
python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/E2/P13/GAC-sweep-s6.json 
python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/E2/P13/GAC-sweep-s9.json 
python scripts/local.py --runs 5 --cpus 1 -e experiments/offline/E2/P13/GAC-sweep-s12.json 