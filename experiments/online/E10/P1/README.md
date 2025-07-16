# Experiment E10 / Phase P1

Evaluating multiple agent types across different plant growth chamber environments with various configurations.

## Environment Types
### CVPlantGrowthChamberColor
Used by Poisson agents (zones 1-6)
#### State
  - time

#### Actions
  - balanced
  - blue
  - red

### MaybeTwilightPlantGrowthChamberDiscrete
Used by Constant agents (zones 10-11)
#### State
  - time

#### Actions
  - light on/off with twilight options

### TimeAreaPlantGrowthChamberDiscrete & TimeDLIPlantGrowthChamberDiscrete
Used by BatchQLearning agents (zones 7-9)
#### State
  - time and additional environment-specific features

#### Actions
  - discrete light control actions

#### Time step
10 minute

## Agents
### Poisson
Stochastic agent with parameter λ=3.0 and max_repeat=6

### Constant
Simple on/off light control with possible twilight periods

### BatchQLearning
Reinforcement learning agent using experience replay and tile coding function approximation

## Deployment
Using alliance chambers zones 1-11
