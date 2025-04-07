#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E6/P0/Spreadsheet.json -i 0
python src/main_real.py -e experiments/online/E6/P0/Spreadsheet.json -i 1
python src/main_real.py -e experiments/online/E6/P0/Spreadsheet.json -i 2
