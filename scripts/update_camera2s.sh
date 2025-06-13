#!/bin/bash

# Define camera2 hostnames in an array
hostnames=(
  "zone01-camera02"
  "zone02-camera02"
  "zone03-camera02"
)

# Loop through each camera hostname and update
for hostname in "${hostnames[@]}"; do
  echo "Updating ${hostname}..."
  rsync -azP api/camera2/ ${hostname}:~/Desktop/camera2
  ssh ${hostname} -t "cd ~/Desktop/camera2 && docker compose pull && docker compose restart"
done
