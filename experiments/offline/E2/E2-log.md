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
- ...
### Conclusions & Outlooks: 
- ...
