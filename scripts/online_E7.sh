#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E7/P2/Constant1.json -i 0
python src/main_real.py -e experiments/online/E7/P2/Constant2.json -i 0
python src/main_real.py -e experiments/online/E7/P3/Bandit3.json -i 0
python src/main_real.py -e experiments/online/E7/P3/ContextualBandit6.json -i 0
python src/main_real.py -e experiments/online/E7/P2/ESARSA8.json -i 0
python src/main_real.py -e experiments/online/E7/P2/ESARSA9.json -i 0
