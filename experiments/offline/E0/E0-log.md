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

##  Phase P3
### Objectives: 
- Remove the averaging of projection factor. See if the 1Relu agent still learns in this more realistic & noisy environment
### Methods: 
- same as P2, but projection factor is now a function of num_steps (instead of clock)
- By the way, all of E0 uses only one plant's data
### Observations: 
- The agent learned, and the learning curves look very similar to (but not the same as) those in P2.
### Conclusions & Outlooks: 
- For realism, it is probably best to use raw projection factor.

##  Phase P4
### Objectives: 
- As per Adam's advice, "we want to give the agent the best shot at learning. To do that we give it as much information as we can", re-introduce observations to the states and check if the agents still learn. 
- Including observation also allows the agents to learn a policy that changes with the plant's developmental stage. (e.g. summer to fall lighting)
### Methods: 
- state := concatenate(clock, current_observed_area / initial_observed_area)
### Observations: 
- Both Relu-DQN agents didn't learn to keep the light on.
### Conclusions & Outlooks: 
- The nn may be too simple. Try LSTM.
- Clock and area are fundamentally different state attributes. The agent has no control over the clock's transition, but the agent's action directly influences the observed area. Try outerproduct instead of concatenate.

##  Phase P5
### Objectives: 
- As per Steven's suggestion, normalize the area observation to between -1 and 1, see if it helps
### Methods: 
- The largest area value from the dataset is 25083.0. Use 30000 as the upperbound; lowerbound is 0.
### Observations: 
- Neither agent learned, but their learning curves look a little better than in P4.
### Conclusions & Outlooks: 
- Normalization could help, but it's probably not the issue.

##  Phase P6
### Objectives: 
- Implement sine time, exponential weighted average (aka moving average), and normalized input in plantsimulator.py
- Train an agent with these improvements
### Methods: 
- following the water treatment plant paper, state := concatenate(sine time, normalized observed area, normalized moving-averaged observed area)
- the areas are normalized to between 0 and 1 by the max value of historic area measurements (in unit of pixels)
- Use 1-day reward function
### Observations: 
- Top three best alphas all converge slowly toward the max return, but don't arrive in 100,000 steps. (too slow!)
### Conclusions & Outlooks: 
- state := concatenate(sine time, normalized observed area, normalized moving-averaged observed area) seems to work well
- using moving-averaged observed area to compute reward seems to help too
- Try 1-step reward

##  Phase P7
### Objectives: 
- Same as P6
### Methods: 
- Same as P6, but using 1-step reward
### Observations: 
- Wow, the performance is terrible, barely better than random policy.
### Conclusions & Outlooks: 
- 1-day reward is vastly better than 1-step reward

##  Phase P8
### Objectives: 
- What happens if we use normalized moving-averaged observed area directly as reward?
### Methods: 
- Same as P6, but reward := normalized moving-averaged observed area 
### Observations: 
- Not good at all.
### Conclusions & Outlooks: 
- 1-day reward is still our current best bet.

