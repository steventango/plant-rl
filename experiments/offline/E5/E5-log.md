#  Experiment E5 -- Sim with only overnight reward
##  Phase P0
### Objectives: 
- Exp 3 with real plants reveal that the only difference in rewards between Constant-Dim and Constant-Bright is the terminal/overnight reward. We should try sim with only overnight reward.
- After group meeting, we decided to stick to 2-week episodes rather than daily episodes. 
- I also think we should make the states more Markov by including trace of actions.
### Methods: 
- Implemented the above change in SimplePlantSimulator and subclass TOD_action.
- state = (linear time-of-day, trace of action)
- reward = daily reward (morning to morning difference), given every morning
- agent = TC ESARSA with replay. TC = tile(time, action trace) -> added a tc strategy called "tc2"
### Observations: 
-
### Conclusions & Outlooks: 
- 