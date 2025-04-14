#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E6/P4/Poisson1.json -i 0
python src/main_real.py -e experiments/online/E6/P4/Poisson2.json -i 0
python src/main_real.py -e experiments/online/E6/P4/Poisson6.json -i 0
python src/main_real.py -e experiments/online/E6/P4/Poisson9.json -i 0
