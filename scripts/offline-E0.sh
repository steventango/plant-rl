#!/bin/sh

# exit script on error
set -e

python src/offline_E0.py -e experiments/offline/E0/DQN-Relu.json -i 0