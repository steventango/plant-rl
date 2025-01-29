#  Experiment E0 -- Train an agent to keep the light on
##  Phase P0
### Objectives: 
- Test out the PlantSimulator environment built from historic plant data
- Train basic DQN agents in PlantSimulator with only two actions [off, on]
### Methods: 
- Try OneLayerRelu and TwoLayerRelu representation networks
- Sweeps over hyperparameter alpha 
- State = concatenate(clock, normalized observed area)
- Reward = difference in normalized observed area between current and previous daytime time stamps
### Observations: 
- Based on the learning curves, the agents did not learn
### Conclusions & Outlooks: 
- The highly fluctuating normalized observed area may be confusing to the agents. 
- If the goal is to learn a daily policy (e.g. circadian rhythm, twilight hours, etc), perhaps only the clock needs to be in the state, as the reward already informs the agent about the observed growth.

##  Phase P1
### Objectives: 
- See if basic DQN agents learn in an updated PlantSimulator where state := clock
### Methods: 
- Same as above except state := clock
### Observations: 
- The learning curves are smoother compared to those in P0.
- If we use the best alpha, it appears that TwoLayerRelu performs better while OneLayerRelu converges to random policy. But OneLayerRelu's non-optimal alpha actually produced a above-random final return. The current performance measure used to select alpha may be unsuitable.
- Still, neither agent managed to get close to the max return for DQN with ep=0.1.
### Conclusions & Outlooks: 
- Using only clock as state seems to improve learning a bit, or at least make learning curves less noisy. 
- Could DQN-2Relu learn if given more steps?
- Consider revising performance measure.
- Try the 24hr reward function

##  Phase P2
### Objectives: 
- See if the agents learn with 24hr reward function
### Methods: 
- state := clock
- reward := (current area - past area) / past area, where past area is the area exactly 24hr prior
- total steps = 100,000
### Observations: 
- Both OneLayerRelu and TwoLayerRelu agents learned very quickly
### Conclusions & Outlooks: 
- This formulation of state & reward is working!