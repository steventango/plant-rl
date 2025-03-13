#!/bin/sh

# exit script on error
set -e

python src/main_real.py -e experiments/online/E5/P0/Spreadsheet.json -i 0
