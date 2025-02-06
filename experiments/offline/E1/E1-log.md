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