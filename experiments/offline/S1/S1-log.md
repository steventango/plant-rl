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
-

### Conclusions & Outlooks:
- 
