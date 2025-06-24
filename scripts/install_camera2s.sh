#!/bin/bash

# Define camera2 hostnames in an array
hostnames=(
  "zone06-camera02"
  "zone08-camera02"
  "zone09-camera02"
)

# Loop through each camera hostname and update in parallel
for hostname in "${hostnames[@]}"; do
  (
    echo "Updating ${hostname}..."
    rsync -azP api/install-docker-bookworm.sh ${hostname}:~/Desktop/
    ssh ${hostname} -t "cd ~/Desktop && ./install-docker-bookworm.sh"
    rsync -azP api/camera2/ ${hostname}:~/Desktop/camera2
    ssh ${hostname} -t "cd ~/Desktop/camera2 && docker compose up -d"
    echo "Update completed for ${hostname}"
  ) &
done

# Wait for all background processes to complete
wait
echo "All camera updates completed"
