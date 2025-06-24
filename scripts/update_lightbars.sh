#!/bin/bash

# Define lightbar hostnames in an array
hostnames=(
  # "alliance-zone1"
  "alliance-zone2"
  # "alliance-zone3"
  # "alliance-zone4"
  # "alliance-zone5"
  # "alliance-zone6"
  # "alliance-zone7"
  # "alliance-zone8"
  # "alliance-zone9"
  # "alliance-zone10"
  # "alliance-zone11"
  # "alliance-zone12"
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
