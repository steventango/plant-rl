#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E7/P0/Spreadsheet1.json -i 0
python src/main_real.py -e experiments/online/E7/P0/Spreadsheet3.json -i 0
