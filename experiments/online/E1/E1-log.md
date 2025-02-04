#  Experiment E0 -- Run a random agent in PlantGrowthChamber
##  Phase P0
### Objectives:
- Test out the PlantGrowthChamber environment
### Methods:
- Random Agent (Uniform Distribution)
### Observations:
- The light color seems to change randomly
### Conclusions & Outlooks:
- Random agent seemingly works.
- Image needs color correction.
- Image needs fish eye correction.
- Agent-environment interaction may not fit the typical RLGlue interface, which assumes
  that that immediately after an action is taken, the environment is in a new state.
  However, what makes more sense is that the new state is recieved after 5 minutes.
  During the delay, the agent should be able to perform planning.
