#!/bin/bash
# Script to get images from all cameras

# Array of zones to turn off
hostnames=(zone01-camera01 zone01-camera02 zone02-camera01 zone02-camera02 zone03-camera01 zone03-camera02 zone06-camera01 zone08-camera01 zone09-camera01)

for hostname in "${hostnames[@]}"; do
  curl http://mitacs-${hostname}.ccis.ualberta.ca:8080/observation --output ${hostname}-observation.jpg
done
