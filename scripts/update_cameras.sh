#!/bin/bash

# Define camera hostnames in an array
hostnames=(
  "zone01-camera01"
  "zone02-camera01"
  "zone03-camera01"
  "zone06-camera01"
  "zone08-camera01"
  "zone09-camera01"
)

# Loop through each camera hostname and update
for hostname in "${hostnames[@]}"; do
  echo "Updating ${hostname}..."
  rsync -azP api/camera/ ${hostname}:~/Desktop/camera
  ssh ${hostname} -t "cd ~/Desktop/camera && docker-compose pull && docker-compose restart"
done
