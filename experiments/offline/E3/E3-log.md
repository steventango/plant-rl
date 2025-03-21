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
- Retrain TC ESARSA in the drastically updated PlantSimulator (smooth ob for both state and reward, horizon countdown, 3 actions). 
### Methods: 
- Use TC because Adam suggested discretized observations
- Pretty wide hyper sweeps
### Observations: 
- Agent tends to stick to one action, either dim or bright. In the second week, some seeds even try to dim during twilight!
- best config
```
representation.tiles:            4
environment.last_day:            14
n_step:                          2
environment.lag:                 1
representation.tilings:          16
environment.trace_decay_rate:    0.9
environment.num_plants:          49
environment.stride:              1
epsilon:                         0.25
alpha:                           0.125
```
### Conclusions & Outlooks: 
- The agent seems to prefer larger exploration! 
- (The following was concluded based on incorrect data analysis) Agent wise, we should try larger n-steps. Smaller epsilon range. and sweep more learning rate. Environment wise, try different trace decay rates like [0, 0.5, 0.9]

##  Phase P2
### Objectives:
- Train TC ESARSA in the further updated PlantSimulator (same as P1 but raw area change as reward, gamma = 1)
### Methods: 
- Reward = 1000 * raw area change over 1 step. 1000 is there just to scale the reward values. Use raw reward so that the agent doesn't game the system (e.g. make the plants move so that they look bigger temperarily). Moreover, with countdown timer in the state, the agent shouldn't be confused by continually increasing raw area change.
- use gamma = 1 because plants' response can be slow, I don't what to discount future rewards. Further, I want the sum of total rewards to be `final area - initial area`
- Use a much finer tile coding so that the area spans ~25 tc grid points over one day.
- different set of hyper sweeps from P1 (check config)
### Observations: 
- It changes up actions more often than P1. Still avoiding off. 
- best config
```
environment.num_plants:          49
environment.trace_decay_rate:    0.9
representation.tiles:            8
environment.lag:                 1
alpha:                           0.003
environment.last_day:            7
representation.tilings:          256
environment.stride:              1
n_step:                          1
epsilon:                         0.1
```
- n_step = 1 being best makes sense in a simulator because plant responds proportionally to lighting change immediately. Real plants might respond in a more complicated way, so it may be more benefitial to use larger n-step. Though it's unclear whether we should use small n_step still (to only capture plant's immediate response to lighting change), or large ones to hopefully capture overnight growth.
### Conclusions & Outlooks: 
- n_step needs to be small in the simulator, but not necessarily in real experiment
- trace_decay_rate will affect things differently in the next updated plantsimulator
- Oliver and I have decided to not use smooth areas in the reward from now on.
- (This was based on incorrect data analysis) epsilon and alpha both needs to be small.

##  Phase P3
### Objectives:
- Train TC-ESARSA again but with different state/reward definitions
### Methods: 
- State = (sin time-of-day, cos time-of-day, sin countdown, cos countdown, average observed area, history of average observed area)
- lagged areas can be safely removed from the state if lag = 1
- Reward = change in average area over 1 step, multiplied by 1000 to scale up (P2 used smooth average area)
### Observations: 
- best config
```
environment.trace_decay_rate:    0.9
environment.last_day:            7
environment.lag:                 1
representation.tilings:          256
environment.num_plants:          48
environment.type:                off_low_high
representation.tiles:            8
environment.stride:              1
epsilon:                         0.05
n_step:                          1
alpha:                           0.0001
```
- More "off" actions than P2, but the last two days look promising.

Just realized that data analysis for P1 and P2 were wrong. After fixing them, I noticed 
- finer tilings didn't cause the "spotty" action history in P3 because P2 shows lots repeated actions. It was likely due to computing reward with unsmoothed areas.
### Conclusions & Outlooks: 
- How to improve upon this?

##  Phase P4
### Objectives:
- Use a smoother history trace decay rate = 0.99, following water treatment plant paper
- Try linear ESARSA for now because I need to modify the tile coding script (too fine of a grid for time!)
### Methods: 
- same as P3
### Observations: 
- linear ESARSA agent learned to keep light either dim or bright the whole time, with some minimal exploration. not bad!
- but it didn't figure out the time-dependent policy
### Conclusions & Outlooks: 
- Adam suggested that adding (i) replay and (ii) Rich's tile coding will help the agent learn the optimal policy.

## Phase P5
### Objectives: 
- Test out the newly updated plantsim (4 discrete actions, unbiased exp moving avg, history of reward rather than of area, removing outliers before averaging areas, using linear time of day, removing countdown timer)
- Test out Rich's tile coding script, with multi call
### Methods: 
- See PlantSimulator_Only1Time
- See representations/RichTileCoder.py
- the tile code is tile(time) + tile(area, Δarea) + tile(time, Δarea)
- Agent = TC ESARSA without replay
### Observations: 
- seed 3 and seed 5 look promising!
### Conclusions & Outlooks: 
- how to improve it?

Between P5 and P6, tested out a few changes that didn't help: 
- increase the freezing time in sim
- give all incorrect actions the same freezing time in sim
- increase lag to 3
- increase stride to 3
- remove penalty during daytime (only freeze growth at night)

What did help:
- using only tile-coded time as state

## Phase P6
### Objectives: 
- Test if incorporating more "time" in tc would help
### Methods: 
- changed up tile coding,  now it's tc = tile(time) + tile(time,area) + tile(time, Δarea)
- Agent = TC ESARSA without replay
### Observations: 
- alpha = 1, ep = 0.05, n_step = 1 does pretty good! almost hitting the optimal policy with seed 1 and 3
- note that hypersweep would select alpha=0.1, which somehow has better return but the action history looks terrible!
### Conclusions & Outlooks: 
- ways to further improve: sarsa(lambda) for better credit assignment, test reward function that focuses on overnight growth, try penalizing inactive weights with L1

