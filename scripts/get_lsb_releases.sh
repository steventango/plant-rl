#!/bin/bash

# Define hostnames in an array
hostnames=(
  "mitacs-zone1"
  "mitacs-zone2"
  "mitacs-zone3"
  "mitacs-zone6"
  "mitacs-zone8"
  "mitacs-zone9"
  "mitacs-zone01-camera01"
  "mitacs-zone01-camera02"
  "mitacs-zone02-camera01"
  "mitacs-zone02-camera02"
  "mitacs-zone03-camera01"
  "mitacs-zone03-camera02"
  "mitacs-zone06-camera01"
  "mitacs-zone06-camera02"
  "mitacs-zone08-camera01"
  "mitacs-zone08-camera02"
  "mitacs-zone09-camera01"
  "mitacs-zone09-camera02"
  "alliance-zone1"
  "alliance-zone2"
  "alliance-zone3"
  "alliance-zone4"
  "alliance-zone5"
  "alliance-zone6"
  "alliance-zone7"
  "alliance-zone8"
  "alliance-zone9"
  "alliance-zone10"
  "alliance-zone11"
  "alliance-zone12"
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

# Loop through each hostname and check lsb_release -a in parallel
for hostname in "${hostnames[@]}"; do
  (
    echo "Checking ${hostname}..."
    ssh ${hostname} -t "lsb_release -a"
  )
done

# Wait for all background processes to complete
wait
