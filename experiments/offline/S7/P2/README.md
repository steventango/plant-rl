# Experiment S7 / Phase P1

Purpose: Obtain an offline RL checkpoint for finetuning in experiment E9/P1. The config is derived from offline/S6/P17/OfflineSARSA.json

## Environment
### PlantGrowthChamber
#### State
  - HOUR
  - DLI

#### Actions
  - dim
  - bright

#### Reward
  - (mean clean plant area from 9:30 am today) - (mean clean plant area from 9:30 am yesterday)

#### Time step
  - 10 minute

## Agents
### BatchESARSA
