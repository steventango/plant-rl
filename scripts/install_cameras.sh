#!/bin/bash

# Define camera hostnames in an array
hostnames=(
  "alliance-zone01-camera01"
  "alliance-zone01-camera02"
  "alliance-zone02-camera01"
  "alliance-zone02-camera02"
  "alliance-zone03-camera01"
  "alliance-zone03-camera02"
  "alliance-zone04-camera01"
  "alliance-zone04-camera02"
  "alliance-zone05-camera01"
  "alliance-zone05-camera02"
  "alliance-zone06-camera01"
  "alliance-zone06-camera02"
  "alliance-zone07-camera01"
  "alliance-zone07-camera02"
  "alliance-zone08-camera01"
  "alliance-zone08-camera02"
  "alliance-zone09-camera01"
  "alliance-zone09-camera02"
  "alliance-zone10-camera01"
  "alliance-zone10-camera02"
  "alliance-zone11-camera01"
  "alliance-zone11-camera02"
  "alliance-zone12-camera01"
  "alliance-zone12-camera02"
)

# Loop through each camera hostname and update in parallel
for hostname in "${hostnames[@]}"; do
  (
    echo "Updating ${hostname}..."
    rsync -azP api/install-docker-buster.sh ${hostname}:~/Desktop/
    ssh ${hostname} -t "cd ~/Desktop && ./install-docker-buster.sh"
    rsync -azP api/camera/ ${hostname}:~/Desktop/camera
    ssh ${hostname} -t "cd ~/Desktop/camera && docker compose up -d"
    echo "Update completed for ${hostname}"
  ) &
done

# Wait for all background processes to complete
wait
echo "All camera updates completed"
