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

##  Phase P5
### Objectives: 
- If we use average area as observation, should we use % change in area as reward, or raw change in area (in unit of pixel)? 
- The latter generally assigns higher rewards to states with larger areas, which may motivate the agent to try to achieve those states? Or is it confusing to the agent that overall the reward increases regardless of the actions?
### Methods: 
- Reward = average area - average lagged area
- The rest same as P3
### Observations: 
- Does much worse than P3, but fairly stable across seeds
### Conclusions & Outlooks: 
- The raw area change is very small because areas are normalized to [0, 1]. Would rescaling the reward help? 

##  Phase P6
### Objectives: 
- See if rescaling reward manually makes P5 viable.
### Methods: 
- Reward = (average area - average lagged area) / 0.02
### Observations: 
- Achieved higher return than P5, but with greater variance across seeds. 
- On average, not nearly as good as P3.
### Conclusions & Outlooks: 
- % change in area works better than raw change in area. 
 
##  Phase P7
### Objectives: 
- Which time step is optimal?
### Methods: 
- State = (time of day, average lagged area, average area)
- Reward = average area / average lagged area - 1
- num_plants = 32, lag = 1 step, stride = [2, 3, 6, 9, 12]
- total run time = 5000 hours
### Observations: 
- As seen in the learning curves, stride = [9, 12] result in near optimal policy within two episodes. The shorter time steps did not reach the optimal policy.
### Conclusions & Outlooks: 
- 90 and 120 minutes, with the former slightly worse as seen in a hiccup near 1000 hours.

##  Phase P8
### Objectives: 
- Now that we know time step is a crucial hyperparameter, could we tune it to make 1-plant environment work?
### Methods: 
- State = (time of day, lagged area, area)
- Reward = area / lagged area - 1
- num_plants = 1, lag = 1 step, stride = [2, 3, 6, 9, 12]
- total run time = 5000 hours
### Observations: 
- Checked that reward history is indeed more noisy with 1 plant only, regardless of the time step.
- Performance is generally worse than P7, and can drop over time. 
### Conclusions & Outlooks: 
- 1-plant environment is significantly harder than 32-plant environment that uses average area. 
- Averaging area really helps stabilize learning.

##  Phase P9
### Objectives: 
- Could `reward = raw change in area` work if we sweep over time-step size? It didn't work in P5, P6.
### Methods: 
- State = (time of day, lagged average area, average area). Note that areas have been normalized to between [0, 1].
- Reward = (area - lagged area) / 0.08. The manual re-scaling by 0.08 is to prevent rewards from being too small, giving this design choice an unfair disadvantage.
- num_plants = 32, lag = 1 step, stride = [2, 3, 6, 9, 12]
- total run time = 5000 hours
- Widen hypersweep of learning rate -- "critic_lr": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
### Observations: 
- Learning curves are all unstable and do not adhere to max return.
- stride = [6, 9, 12] reached near max return after two episodes, but quickly collapsed.
### Conclusions & Outlooks: 
- % area change still seems to be better. 
 
##  Phase P10
### Objectives: 
- Oliver realized an error in GAC's code which set gamma=0 this whole time! This made the agent very myopic.
- How would GAC perform with gamma = 0.99?
### Methods: 
-
### Observations: 
- 
### Conclusions & Outlooks: 
- 

 