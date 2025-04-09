#!/bin/sh

# exit script on error
set -e

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 13:30' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet1.json
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet2.json
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet6.json
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet9.json
