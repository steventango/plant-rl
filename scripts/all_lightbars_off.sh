#!/bin/bash
# Script to turn off lights in specified zones

# Array of zones to turn off
hostnames=(
  # mitacs-zone1.ccis.ualberta.ca
  # mitacs-zone2.ccis.ualberta.ca
  # mitacs-zone3.ccis.ualberta.ca
  # mitacs-zone6.ccis.ualberta.ca
  # mitacs-zone8.ccis.ualberta.ca
  # mitacs-zone9.ccis.ualberta.ca
  42.244.191.26
  42.244.191.27
  42.244.191.28
  42.244.191.29
  42.244.191.30
  42.244.191.31
  42.244.191.32
  42.244.191.33
  42.244.191.34
  42.244.191.35
  42.244.191.36
  42.244.191.37
)

for hostname in "${hostnames[@]}"; do
  curl http://$hostname:8080/action -X PUT -H "Content-Type: application/json" -d '{"array": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}' &
done
