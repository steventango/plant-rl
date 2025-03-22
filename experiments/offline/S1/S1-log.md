#  Experiment S1 -- PlantSimulator Challenges

## Phase P0
### Objectives:
- Test out the reward function that is the difference of two exponential moving averages of the area of the plants.
### Methods:
- See only1time_emareward
- See representations/RichTileCoder.py
- Agent = TC ESARSA without replay
### Observations:
- The way the original reward function is structured, there is a temporal credit assignment problem.
- This modified reward function also doesn't work.

### Conclusions & Outlooks:
- Try SARSA(λ) instead.

## Phase P1
### Objectives:
- Try ESARSA(λ) with the original reward function.
### Methods:
- See only1time
- See representations/RichTileCoder.py
- Agent = TC ESARSA without replay
### Observations:
- Works better, but maybe generalization across the time dimension is not working as well.

### Conclusions & Outlooks:
- Adjust the time representation.

## Phase P2
### Objectives:
- Try ESARSA(λ) with the original reward function, but with only the current time step as input.
### Methods:
- See onlytime
- See representations/RichTileCoder.py
- Agent = TC ESARSA without replay
### Observations:
- I tried many things and some of they contributed to something working.
  - state aggregation instead of tiling
  - hyper selection using return of last 20% of timesteps
  - forcing epsilon=0.25
  - The total returns / the return of the last 20% of timesteps,  favour a constant policy over our learned policy that visually does better
### Conclusions & Outlooks:
- Address issues with total returns.
- We need to run an ablation study to understand what made it work.
