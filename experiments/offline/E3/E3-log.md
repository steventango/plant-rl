#  Experiment E3 -- PlantSimulator Challenges
##  Phase P0
### Objectives: 
- Can TC or linear ESARSA beat binary plant simulator in 1 day?
### Methods: 
- Use 49 plants
- Train for 1 plant day
- Pick best hypers based on AUC in the action history
- lag = stride = 1
### Observations: 
- They both kind of worked
### Conclusions & Outlooks: 
- Simple ESARSA seems to learn light-on is good even with time step = 10min.

##  Phase P1
### Objectives:
- Retrain TC ESARSA in the drastically updated PlantSimulator (smooth ob, horizon countdown, 3 actions). 
### Methods: 
- Use TC because Adam suggested discretized observations
- Pretty wide hyper sweeps
### Observations: 
- TC ESARSA did not learn. 
### Conclusions & Outlooks: 
- How to make it work? Agent wise, we should try larger n-steps. Smaller epsilon range. and sweep more learning rate. Environment wise, try different trace decay rates like [0, 0.5, 0.9]

##  Phase P2
### Objectives:
- Train TC ESARSA in the further updated PlantSimulator (same as P2 but raw area change as reward, gamma = 1)
### Methods: 
- Reward = 1000 * raw area change over 1 step. 1000 is there just to scale the reward values. Use raw reward so that the agent doesn't game the system (e.g. make the plants move so that they look bigger temperarily). Moreover, with countdown timer in the state, the agent shouldn't be confused by continually increasing raw area change.
- use gamma = 1 because plants' response can be slow, I don't what to discount future rewards. Further, I want the sum of total rewards to be `final area - initial area`
- Use a much finer tile coding so that the area spans ~25 tc grid points over one day.
- different set of hyper sweeps from P2 (check config)
### Observations: 
- 
### Conclusions & Outlooks: 
- 

