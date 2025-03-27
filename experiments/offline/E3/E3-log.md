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

## Phase P7
### Objectives: 
- Test all the updates
### Methods: 
- Updates to PlantSimulator: "get_observation" method, new upperbound for area normalization (agreed upon with Steven), modularized iqm 
- Updates to TC ESARSA: Steven added eligibility trace lambda, tc = tile(time) + tile(time,area) + tile(time, Δarea) + tile(area,Δarea), optimistic initial value w0=1.0
### Observations: 
- for lambda=0.75, alpha=0.01, ep = 0.1, one of the seed demonstrates ability to learn time-dependent policy, but seem to have problem at the overnight transition.
### Conclusions & Outlooks: 
- Up until now, I consider the reward as a combination of (i) reward from incremental change and (ii) reward from overnight growth, but the latter is in fact not very compatible with a 24hr agent. In this experiment, there is a slight evidence that the large overnight signal is not benefitial.
- what if the reward is only based on incremental change? A good reward would then be contributed by (i) incremental growth and/or (ii) incremental leave opening. The former is definitely legit. The latter is also something we want to optimize anyways, because according to Glen, wide open leaves is a good sign and leads to growth.
- Of course there is a limit on how much the plants can stretch wide open and grow, beyond this limit the plant may die. The agent will have to learn this limit itself. An analogy is the garbage picking robot, how long should it roam around before risking a dead battery?
- Since raw change in area per time step is VERY noisy, I think it's sensible to use its trace.

## Phase P8
### Objectives: 
- A new way of thinking this task: our agent's only job is to maximize incremental change in observed area. 
- The developmental stage of the plant no longer matters. The agent is always adjusting its lighting policy anyways (online learning of a nonstationary plant preference). So we can remove any state inputs that tend to grow. 
- let Reward = normalized trace(% change in average area) = last state input
### Methods: 
- State = (linear time-of-day, history of % change in average area), both inputs are normalized. The latter is clipped to [0,1] because the trace acts wildly at the beginning of the run.
- Reward = second state input
- Tile code = tile(2d state space). 
### Observations: 
- not bad. see plot
### Conclusions & Outlooks: 
- I noticed that the first day of reward is basically a mess due to trace. Maybe we should nullify it.

## Phase P9
### Objectives: 
- see if it's helpful to remove day 1's reward, which is senseless.
### Methods: 
- same as above
- try lambda = 0.9, 0.1, 0.0
- make sure w0 = 0 so that the initial r=0 simply doesn't update the net
### Observations: 
- seed 3 is pretty darn good. best config: alpha=0.3, ep=0.1, lambda=0. lambda=0 is sensible in sim, but real exp should use large lambda=0.9.
### Conclusions & Outlooks: 
- the improvements above are pretty good!

## Phase P10
### Objectives: 
- Test out a few reasonable changes (see methods)
### Methods: 
- mostly same as above
- in sim, modified the percentage change formula slightly so that an up-and-down peak in area doesn't result in a net positive return (so called symmetric percentage change)
- tile coding = tile(time) + tile(trace) + tile(time, trace).
- w0 doesn't have to be zero, because day 1's trace and reward (which are messy) have been set to 0
### Observations: 
- Not as good as P9. Could it be because it's harder to learn with the longer feature vector?
### Conclusions & Outlooks: 
- Try L1 regularization to penalize dead weights

## Phase P11
### Objectives: 
- Test out L1 regularization
### Methods: 
- added L1 regularization to TC ESARSA
- sweep through "l1": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
### Observations: 
- The agent seems to converge to flat line policy. Why can't it do any better? Is it not expressive enough? Maybe the 2D tile code gets penalized to death because it has 32 features, but the 1D tile codes each also has 32 features. 
### Conclusions & Outlooks: 
- Try using only 2Dtile = (time, growth rate) with L1 regularization

## Phase P12
### Objectives: 
- Test out 2Dtile = (time, growth rate) with L1 regularization. 
### Methods: 
- Basically P9 but with symmetric % change in sim and L1 reg in agent.
### Observations: 
- ...
### Conclusions & Outlooks: 
- ..