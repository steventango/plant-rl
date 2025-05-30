#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/offline/E6/P0/tc-ESARSA.json -i 0
python scripts/local.py --runs 30 --cpus 18 -e experiments/offline/E6/P1/ESARSA.json --entry "src/main.py --silent"
python src/main_real.py -e experiments/offline/E6/P2/tc-ESARSA.json -i 0
python src/main_real.py -e experiments/offline/E6/P2/Poisson-slow.json -i 0
python src/main_real.py -e experiments/offline/E6/P2/Poisson.json -i 0

python src/main_real.py -e experiments/offline/E6/P3/Spreadsheet1.json -i 0
python src/main_real.py -e experiments/offline/E6/P3/Spreadsheet2.json -i 0
python src/main_real.py -e experiments/offline/E6/P3/Spreadsheet6.json -i 0
python src/main_real.py -e experiments/offline/E6/P3/Spreadsheet9.json -i 0
