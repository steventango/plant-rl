#!/bin/sh

# exit script on error
set -e

current_epoch=$(date +%s.%N)
target_epoch=$(date -d "next day 00:00:00.0" +%s.%N)
sleep_seconds=$(echo "$target_epoch - $current_epoch"|bc)
sleep_minutes=$(echo "$sleep_seconds / 60"|bc)

echo "Sleeping for $sleep_minutes minutes"
sleep $sleep_seconds
python src/main_real.py -e experiments/online/E4/P0/Spreadsheet.json -i 0
