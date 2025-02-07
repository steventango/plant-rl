#  Experiment E1 -- Which agents can beat PlantSimulator?
##  Phase P0
### Objectives: 
- See if a DQN or SoftmaxAC agent can beat the basic version of PlantSimulator (with two actions [off, on])
### Methods: 
- TwoLayerRelu with 32 neurons
- SoftmaxAC with greedy tau=[0.001, 0.01, 0.1] & myopic n_step=1 
(btw had to first normalize sine & cosine times to between [0, 1] for the tile coder)
- SoftmaxAC with greedy tau=[0.001, 0.01] & farsighted n_step=16
### Observations: 
(in order)
- About the same as, if not worse than, OneLayerRelu
- Didn't learn at all, but greedier tau=[0.001, 0.01] seemed to work better with matching alpha=[0.001, 0.01]
- Greedy & farsighted SoftmaxAC achieved max return at ~70,000 steps. Best params: tau=0.01, alpha=0.03, n_step=16
- Note: noticed that {"episode_cutoff": -1} means no cutoff.
### Conclusions & Outlooks: 
- Greedy & farsighted SoftmaxAC beat basic PlantSimulator, albeit after a long time (~2.7 years)
- Adam suggested to add experience replay and that GreedyAC will likely do better.
- Oliver and I think we need some sort of RNN in the agent, because the environment is partially observable and the current reward is affected by past actions.

## Phase P1
### Objectives: 
- As per Oliver's suggestion, include 24hr-prior-area in the state.
- It makes sense to include 24hr-prior-area, because it's used to compute the reward
### Methods: 
- State = concatenate(sin time, cos time, area, smoothed area, 24hr-prior area, smoothed 24hr-prior area)
- Change decay rate from 0.9 to 0.99, as per the paper "GVFs in the real world..."
- Use raw area in Reward: (area - 24hr-prior area) / 24hr-prior area
- SoftmaxAC with greedy tau=[0.001, 0.01] & farsighted n_step=[8, 16]
### Observations: 
- Didn't converge to max return, but if I removed the two smoothed areas, the agent nearly learned at 100,000 steps
- It seems that the less state information I give the agent, the better it performs. In E0, DQN learned rapidly when state = clock. Maybe with a higher-dimensional state space, there is not enough experiences to learn meaningful policy and value function.
### Conclusions & Outlooks: 
- How do we gather more experiences and reuse them? Consider treating each plant separately.
- Use neural net actor-critic, rather than tile coding, to help generalize across states.