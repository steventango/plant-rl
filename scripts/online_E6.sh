#!/bin/sh

# exit script on error
set -e

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 17:00' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P3/Spreadsheet1.json -i 0

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 17:00' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P3/Spreadsheet2.json -i 0

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 17:00' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P3/Spreadsheet6.json -i 0

current_epoch=$(date +%s)
target_epoch=$(date -d '04/09/2025 17:00' +%s)
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E6/P3/Spreadsheet9.json -i 0
