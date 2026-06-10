#!/bin/bash

# Define lightbar hostnames in an array
hostnames=(
  "zone01"
  "zone02"
  "zone03"
  "zone04"
  "zone05"
  "zone06"
  "zone07"
  "zone08"
  "zone09"
  "zone10"
  "zone11"
  "zone12"
)

# Loop through each lightbar hostname and update in parallel
for hostname in "${hostnames[@]}"; do
  (
    echo "Updating ${hostname}..."
    rsync -azP api/lightbar/ ${hostname}:~/Desktop/lightbar
    ssh ${hostname} -t "cd ~/Desktop/lightbar && docker compose pull && docker compose restart"
    echo "Update completed for ${hostname}"
  )
done

# Wait for all background processes to complete
wait
echo "All lightbar updates completed"
