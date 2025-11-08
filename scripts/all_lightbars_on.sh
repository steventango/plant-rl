#!/bin/bash
# Script to turn on lights in specified zones

# Array of zones to turn on
hostnames=(
  # mitacs-zone1.ccis.ualberta.ca
  # mitacs-zone2.ccis.ualberta.ca
  # mitacs-zone3.ccis.ualberta.ca
  # mitacs-zone6.ccis.ualberta.ca
  # mitacs-zone8.ccis.ualberta.ca
  # mitacs-zone9.ccis.ualberta.ca
  142.244.191.26
  142.244.191.27
  142.244.191.28
  142.244.191.29
  142.244.191.30
  142.244.191.31
  142.244.191.32
  142.244.191.33
  142.244.191.34
  142.244.191.35
  142.244.191.36
  142.244.191.37
)

for hostname in "${hostnames[@]}"; do
  curl http://$hostname:8080/action -X PUT -H "Content-Type: application/json" -d '{"array": [[0.398, 0.762, 0.324, 0.000, 0.332, 0.606], [0.398, 0.762, 0.324, 0.000, 0.332, 0.606]]}' &
done
