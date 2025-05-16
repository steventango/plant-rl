#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/offline/S2/P2/Bandit.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P2/ContextualBandit.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P2/ESARSA-A.json -i 0 &
python src/main_real.py -e experiments/offline/S2/P2/ESARSA-B.json -i 0 &
wait
