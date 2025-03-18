#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 1 -e experiments/online/E5/P0/Spreadsheet.json --entry src/main_real.py
