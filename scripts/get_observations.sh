#!/bin/bash
# Script to get images from all cameras

# Array of zones to turn capture images from
hostnames=(
  mitacs-zone01-camera01
  mitacs-zone01-camera02
  mitacs-zone02-camera01
  mitacs-zone02-camera02
  mitacs-zone03-camera01
  mitacs-zone03-camera02
  mitacs-zone06-camera02
  mitacs-zone08-camera02
  mitacs-zone09-camera02
  alliance-zone01-camera01
  alliance-zone01-camera02
  alliance-zone02-camera01
  alliance-zone02-camera02
  alliance-zone03-camera01
  alliance-zone03-camera02
  alliance-zone04-camera01
  alliance-zone04-camera02
  alliance-zone05-camera01
  alliance-zone05-camera02
  alliance-zone06-camera01
  alliance-zone06-camera02
  alliance-zone07-camera01
  alliance-zone07-camera02
  alliance-zone08-camera01
  alliance-zone08-camera02
  alliance-zone09-camera01
  alliance-zone09-camera02
  alliance-zone10-camera01
  alliance-zone10-camera02
  alliance-zone11-camera01
  alliance-zone11-camera02
  alliance-zone12-camera01
  alliance-zone12-camera02
)

for hostname in "${hostnames[@]}"; do
  curl http://${hostname}.ccis.ualberta.ca:8080/observation --output ${hostname}.jpg &
done
wait
