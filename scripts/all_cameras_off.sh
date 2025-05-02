#!/bin/bash
# Script to shut down all cameras using SSH

# Array of camera hostnames
hostnames=(
  zone01-camera01
  zone01-camera02
  zone02-camera01
  zone02-camera02
  zone03-camera01
  zone03-camera02
  zone06-camera01
  zone08-camera01
  zone09-camera01
)

for hostname in "${hostnames[@]}"; do
  echo "Shutting down camera: ${hostname}"
  # SSH into each camera and execute shutdown command
  ssh -t ${hostname} "sudo shutdown -h now"
done

echo "All cameras shutdown commands sent"
