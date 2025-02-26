#  Experiment E2 -- Improve PlantSimulator
##  Phase P0
### Objectives: 
- Does averaging the rewards of individual plants help reduce noises in reward?
### Methods: 
- Method1: individual reward = area / lagged_area - 1; env.reward = average(individual rewards of 32 plants)
- Method2: averaged area = average of 32 observed areas. env.reward = ave_area / lagged_ave_area - 1
### Observations: 
- Both Method1 and Method 2 reduce the noise by about the same amount (half of the noise when using only one plant).
- If using Method 2 but with median instead of average, noise is less reduced.
- If using Method 1 but env.reward = np.percentile(individual rewards, 20), the noise is reduced as if we are using mean, but the difference between the overnight rewards of good and bad policies becomes more obvious.
### Conclusions & Outlooks: 
- Using the average area of multiple plants reduces the noise magnitude in the reward.
- It is even better to use the 20th percentile.

##  Phase P1
### Objectives: 
- Can GAC learn to keep the light on with long state vector := (time of day, lagged observed areas, observed areas) AND 1-step reward? 
### Methods: 
- Reward = np.percentile(1-step rewards, 20) 
- num_plants = 32, which makes the state vector 66 entries long
- Hypersweep GAC's params
### Observations: 
- The performance is not too bad, considering the agents are given noisy 1-step reward. Definitely not hitting the optimal policy though.
### Conclusions & Outlooks: 
- Maybe the state vectors are too long?

##  Phase P2
### Objectives: 
- Will GAC learn better if we ignore individual plants and simply use the average area?
- We keep the challenge posted by noisy 1-step reward.
### Methods: 
- State = (time of day, average lagged area, average area)
- Reward = average area / average lagged area - 1
- num_plants = 32, lag = 1 step
- Hypersweep GAC's params
- Use --cpus 1 as per Steven: "something about GreedyAC scales poorly with multiple processes. It seems to run faster sequentially than in parallel."
- Edited plotting script in P1, P2 so that hyper params are selected by AUC
### Observations: 
- Agent didn't learn. The performance is about the same as P1, but with more variance across seeds.
- Change in plotting script doesn't result in a different best config.
### Conclusions & Outlooks: 
- Reducing the length of state vector didn't help. The problem is else where. However since it doesn't matter, I will stick with the smaller state vector for simplicity.

##  Phase P3
### Objectives: 
- Play with the size of time step. 
- Can GAC learn if time step is longer, say 1hr (i.e. only 12 actions/day as opposed to 72)?
### Methods: 
- Added a "rescale" method in utils.functions.PiecewiseLinear
- Added a new parameter to MultiPlantSimulator, "stride", such that env time step = stride * spreadsheet time step 
- stride = 6, otherwise same as P2
### Observations: 
- It learned really good! But it falls short of optimal policy. Is it due to exploration?
### Conclusions & Outlooks: 
- Changing time scale really helepd!

##  Phase P4
### Objectives: 
- Do a smoke test as per Adam's suggestion
### Methods: 
- Added a new class SmokeTest_MultiPlantSimulator that manually sets areas to near zero when action = off. 
- Reward = average area
- The rest same as P3
### Observations: 
- Observed that if reward = average area / average lagged area - 1, the reward is SO HUGE after an off-to-on transition that the return of a random policy is better than the light-on policy. So here we use reward = average area
- Agent learned the optimal policy after 2 episodes.
### Conclusions & Outlooks: 
- Smoke test worked
 