#  Experiment E4 -- Macro and micro trends in observations
##  Phase P0
### Objectives: 
- Reward should capture the macro trend in observation, which is the overall growth of the plants.   - The micro trend that captures plants' behavior can be in the state so that agent can make adjustments accordingly to maximize reward.
- Test out the above setup in sim
### Methods: 
- env = SimplePlantSimulator
- state = (linear time-of-day, trace of % change in average area, trace of average area) -> revert % change to nonsymmetric one because micro trace is no longer used for reward
- reward = trace of average area = state[-1]
- agent = TC ESARSA. TC = tile(time, micro trace) + tile(macro trace)
### Observations: 
- Agent tends to stick with dim or med. At least it behaves pretty consistently across seeds (except seed5)
### Conclusions & Outlooks: 
- How to improve? Try a sparse version of the reward as Adam previously suggested.

##  Phase P1
### Objectives: 
- Test out sparse reward
### Methods: 
- Reward = % change of macro trace over 24hr, given only at the end of day
### Observations: 
- not terrible but not great
### Conclusions & Outlooks: 
- What if we hand out "% change of macro trace over 24hr" at every time step?

##  Phase P2
### Objectives: 
- Test out a dense reward that is "normalized" (i.e. not going up and up), in contrast to P0
### Methods: 
- Reward = % change of macro trace over 24hr, given at every time step 
### Observations: 
- Performance is similar to P0: agent converges too fast to non-optimal policy. 
### Conclusions & Outlooks: 
- Sparse reward seems to work better. Can try more frequent sparse reward to speed up learning.

##  Phase P3
### Objectives: 
- Try a frequent sparse reward: reward every 90 minutes.
### Methods: 
- make optimal policy 1 if during first and last two hours of the day, 2 otherwise
- Implement reward_freq. Sweep over [1, 9, 18, 36, 72]
- Analyze data for each freq separately because their rewards are not comparable.
### Observations: 
- For alpha=0.1, freq=72 is the only agent that self corrects from a bad policy to a decent policy (see seed 3). Agents with other freq would flat line on a bad policy.
- alpha=1.0 seems to be too high, so the noise in the observation adds much randomness to the policy. 
### Conclusions & Outlooks: 
- Daily reward works best compared to more frequent sparse rewards.

##  Phase P4
### Objectives: 
- Oliver raised a legit concern that daily reward is too sparse for a 7-day run. 
- Try shorter reward period, and smoother macro trace (beta=0.995) to remove modulation.
### Methods: 
- Added a 1D tile of time in the tile coding because it makes sense to generalize the policy to the same time of day, regardless of micro trace
- increase beta of macro trace from 0.99 to 0.995 (note: accidentally reverted back to 0.99 for reward period >=12)
### Observations: 
- inconsistent performance across seeds and different reward periods.
### Conclusions & Outlooks: 
- Something is still not working.

##  Phase P5
### Objectives: 
- Instead of moving average, try average
### Methods: 
- Added a sparse_reward_function_avg that takes the average during every reward period and compute its percentage growth above the average during the previous reward period
- Still keep the macro trace in the state with beta = 0.99
- fixed a bug in the tile coder (the 1dtile(time) was not wrapped)
### Observations: 
- Using averaging reveals that random policy and optimal policy in fact lead to very similar rewards. This is a fundamental challenge with plant RL, not due to poor reward design. In fact, a poor reward design is one that artificially accentuates the difference between these two policies, by over assigning credits. For example, for sparse_reward_function_uema, the plants' entire growth trajectory is kept in uema, so using uema to compute growth over one reward period tends to overstimate/underestimate the reward if the plant has always been doing well/poorly.
- It seems most logical to average over 24hr reward period, because plants' motion due to its circadian rhythm is 24h-periodic. This calls for agents with replay to learn from the sparse rewards.
- Interestingly, all 5 seeds converges to flat line policy at 2, which is the best flat line policy. 
### Conclusions & Outlooks: 
- Average is definitely a better choice than moving average when it comes to assessing growth over a period.

##  Phase P6
### Objectives: 
- Plan Exp 4 for real plants
### Methods: 
Sim: 
- the optimal policy is now 2 at all times. All other actions are penalized by stalling growth by a full time step.
- the state includes time of day, average area, and percentage change in average area over 1 time step, excluding the overnight step (because the purpose of this state input is to track plant motion)
- the reward is raw change in avg area, which includes overnight growth.
- the reward is normalized to be roughly in [0,1]
Agent: 
- ESARSA(0)
- tile coder is editted so that strategy = "tc1" is now tc = tile(time) + tile(time, plant motion) + tile(plant area) and "tc2" is removed
- optimistic initialization w0 = 1 / num_nonzero_features
- use rougher tc grid.
### Observations: 
- This set up works pretty well. All 5 seeds converges to 2.
- Rougher tiling works better (tilings = 8 works better than 16)
- alpha = 0.3 is best
- Noted from earlier compute (which has been overwritten) that optimistic initialization makes very little difference in agent's action history. Steven suggested that the difference could be more obvious if the agent were trained for longer.
### Conclusions & Outlooks: 
- The ESARSA(0) agent was able to beat the simplified challenge of finding the optimal action out of 4 choices. 
- I think the main problem with the previous attempts was that the tile coding was too fine, leading to poor state coverage.

##  Phase P7
### Objectives: 
- Test out bandit agent
### Methods: 
Sim: 
- Added a subclass to SimplePlantSimulator s.t. there is only one state, gamma = 0, and overnight reward is set to 0.
Agent: 
- Use tc-ESARSA with only 1 state
- lambda = 0
### Observations: 
- Plotted the bandit's value function using imshow. vmin=-0.03, vmax=0.03. Observe that the agents do converge to action 2.
- During the last few days, the value of action 2 decreases towards the end of each day. This is because the reward goes down as the plants close up (this closing motion is pronounced in Curtis's old data).
### Conclusions & Outlooks: 
- Bandit agent works as expected. It learned that 2 is best but the value of action 2 decreases when the plants are closing.

##  Phase P8
### Objectives: 
- Implement daily episode in simpleplantsim
- Try both bandit and contextual bandit
### Methods: 
Sim: 
- Added a subclass that starts a new episode everyday 
- Added one final data point to complete the 14th day
- Replaced old data with Steven's reprocessed "clean_area"
Agent: 
- Use tc-ESARSA with only 1 state
- lambda = 0
### Observations: 
- Plotted the bandit's value function using imshow. vmin=-0.03, vmax=0.03. Observe that the agents do converge to action 2.
- During the last few days, the value of action 2 decreases towards the end of each day. This is because the reward goes down as the plants close up (this closing motion is pronounced in Curtis's old data).
### Conclusions & Outlooks: 
- Bandit agent works as expected. It learned that 2 is best but the value of action 2 decreases when the plants are closing.
- Try contextual bandit


