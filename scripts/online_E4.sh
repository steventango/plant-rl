#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E4/P1/Spreadsheet.json -i 0
