#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/offline/S2/P1/Bandit.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P1/ContextualBandit.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P1/Constant-DIM.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P1/Constant-ON.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P1/ESARSA-A.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P1/ESARSA-B.json -i 0 &
wait
