# Experiment E6 / Phase P1

Collect data using Poisson agents.

## Environment
### PlantGrowthChamber
#### State
  - time

#### Actions
  - moonlight
  - low
  - medium
  - high

#### Time step
5 minute

## Agents
### Poisson
Action ~ Uniform(4)
Repeat ~ (5 minutes) * (Poisson(2) + 1)
