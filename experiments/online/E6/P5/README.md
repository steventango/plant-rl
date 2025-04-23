# Experiment E6 / Phase P5

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
  - blue low
  - blue medium
  - blue high
  - red low
  - red medium
  - red high

#### Time step
10 minute

## Agents
### Poisson
Action ~ Uniform(10)
Repeat ~ (10 minutes) * (Poisson(2) + 1)
