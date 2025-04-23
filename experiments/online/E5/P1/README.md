# Experiment E5 / Phase P1

## Environment
### CVPlantGrowthChamberDiscrete
#### State
  - normalized time of day
  - normalized unbiased exponential moving average of change in IQM area

#### Actions
  - off
  - low
  - medium
  - high

#### Time step
5 minute

## Agents
### Poisson
Action ~ Uniform(4)
Repeat ~ (5 minutes) * (Poisson(2) + 1)

### Poisson-slow
Action ~ Uniform(4)
Repeat ~ (5 minutes) * (Poisson(3) + 1)

## ESARSA
Tile coding
