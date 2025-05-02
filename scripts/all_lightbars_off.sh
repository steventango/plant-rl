#!/bin/bash
# Script to turn off lights in specified zones

# Array of zones to turn off
zones=(1 2 3 6 8 9)

for zone in "${zones[@]}"; do
  curl http://mitacs-zone$zone.ccis.ualberta.ca:8080/action -X PUT -H "Content-Type: application/json" -d '{"array": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}' &
done
