# Experiment S7 / Phase P0

Purpose: Obtain an offline RL checkpoint for finetuning in experiment E9/P1. The config is derived from offline/S6/P17/OfflineSARSA.json

## Environment
### PlantGrowthChamber
#### State
  - HOUR
  - AREA

#### Actions
  - dim
  - bright

#### Reward
  - change in mean clean plant area

#### Time step
  - 10 minute

## Agents
### BatchESARSA
