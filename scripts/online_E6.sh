#!/bin/sh

# exit script on error
set -e

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 13:40' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet1.json -i 0

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 13:40' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet2.json -i 0

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 13:40' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet6.json -i 0

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 13:40' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P2/Spreadsheet9.json -i 0
