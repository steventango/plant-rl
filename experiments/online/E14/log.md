# Experiment description by zones
1. standard white lighting
2. deterministic policy trained on offline dataset, no online learning
3. stochastic policy trained on offline dataset, no online learning
4. stochastic policy trained on offline dataset, no online learning
5. stochastic policy trained on offline dataset, no online learning
6. stochastic policy trained on offline dataset, no online learning
7. None
8. deterministic policy trained on GP, no online learning, optimism = 0
9. deterministic policy trained on GP, no online learning, optimism = 0.25
10. deterministic policy trained on GP, no online learning, optimism = 0.5
11. deterministic policy trained on GP, no online learning, optimism = 0.75
12. deterministic policy trained on GP, no online learning, optimism = 1

# Notes:
## Day 1 2025-11-12
- All agents are InAC. Zones 2-6 agents were trained by data replay. Zones 8 - 12 agents were trained by GP sim.
- State = (day, area, one-hot action, one-hot action trace with beta=0.9)
Action = (red coeff, white coeff, blue coeff), intensity fixed at 105 ppfd
Reward as usual 
- optimism means, when training in GP sim, the "area" in the next state is predicted_mean + optimism * predicted_stdev. So if optimism = 0, it predicts the mean (on average, because GPsim returns a sampled value)
- All InAC agents use the default hypers + weight decay=1e-4.
- Agents trained on data are trained for 1 M  updates.
Agents trained in sim are trained for 1 k GP simulator steps, 100k updates, for some reason this led to better performance when evaluated in the simulator.
- there were a lot of unexpected technical difficulties this time: "X11 forwarding for zone clicking, nvidia drivers, grounding dino and segment anything", which Steven worked through the night before the experiment, thankfully
- today the laptop hardware is failing, with both the battery and power cable malfunctioning. we're gonna try to replace them | the issue may have been fixed by a change in BIOS (switching the power settings from adaptive to primarily AC) | also ordered a backup power adaptor just in case, cuz the cable is shredding on the current one | we need that new computer sooner than later.
- The first 3 hours of data from this morning have been impacted by the battery issue, a checkpointing issue, and wandb network connectivity issues. The main issue is that restarts caused the agent wrapper to lose its state and execute a different action (see figures below). These issues were fixed by 11:36 AM. We also lost the first 20 minutes of twilight. The data for today may still be useable if we assume the agent took the mean of all the actions taken between 9:30 and 20:30 today.
- malfunctioning action of a stochastic InAC
![alt text](image.png)
- malfunctioning action of a deterministic agent
![alt text](image-1.png)

## Day 3 2025-11-14
- Noticed that more than half of the deterministic chambers were favoring blue
- Noticed that in the deterministic chambers, about 5-7 out of 36 plants are dead (for now we're only using the 36 plants in the middle region for the CV). Maybe because we CS people helped with planting, the survival rate dropped. 
- Instead of using the mean of plant areas in the state, we decided to use iqm. We set the iqm to take the values between quantile = 0.25 and 0.9, and average them. 
- At 9:10pm - 9:45pm, redeployed all the agents (except zone1) with the iqm change. Noted that it took about 135 seconds to docker compose down completely. All zones loaded checkpoints properly. However, zone 8 exceeded time limit when docker down; hopefully the checkpoint it was saving was indeed saved properly. Could check zone 8's action trace tomorrow.

## Day 4 2025-11-15
Action choices over the last four days: (red, white, blue) coeffs
- zone2 
[0.33 0.34 0.33]
[0.6  0.03 0.37]
[0.5 0.  0.5]
[0.4  0.11 0.49]
- zone8
[0.05 0.69 0.26]
[-0.   0.5  0.5]
[-0.   0.5  0.5]
[-0.   0.5  0.5]
- zone9 
[0.14 0.23 0.63]
[-0.   0.5  0.5]
[-0.   0.5  0.5]
[-0.   0.5  0.5]
- zone10
[0.11 0.38 0.52]
[-0.   0.5  0.5]
[-0.   0.5  0.5]
[-0.   0.5  0.5]
- zone11
[0.06 0.58 0.36]
[0.33 0.33 0.33]
[0.33 0.33 0.33]
[0.33 0.33 0.33]
- zone12
[0.04 0.35 0.61]
[0.5 0.5 0. ]
[0.5 0.5 0. ]
[0.5 0.5 0. ]
- for the GP trained agents, the action choices are suspiciously located at special points on the simplex, which may suggest bugs
- Found bug #1: the param "normalize" should be true in the InAC*.json for * = 8 to 12. need to redeploy those zones by directly modifying the env
- Found bug #2: the trace in the dataset used to trained GP model has beta=0.1/alpha=0.9, instead of beta=0.9/alpha=0.1 as planned. Then GP sim uses trace with beta=0.9/alpha=0.1. Then the PlantGrowthChamber env had beta=0.1/alpha=0.9 again! 
- Bug #1 can be fixed relatively easily. But Bug #2 can't be fixed because the action traces have been initialized already. If we want to reinitialize with a different alpha, we will have to delete the checkpoints. 
- Steven will redeploy zones 8-12 tonight with normalize set to true. But we will leave the wrong traces alone and hope that the agents will be able to generalize to unseen traces, and choose some interesting actions beyond the special points. 
- Zones 8 - 12 are basically write offs, but the data can be useful for training CV. 
- (9:41pm) Steven: "i gave up on trying to save the action trace, so it just got rebooted with correct date, normalization, α trace α, but the old action trace is gone. should be fine since the traces are adjusted they’ll be roughly normal in a few days"

## Day 6 2025-11-17
- Steven made changes to our codebase so that our environments take beta as a param (the uema script remains unchanged)
- Need to think about what to do when the dirichlet distribution has at least one alpha <= 1. In this case mode() returns nan and the script falls back to using mean() for the next action. Does it make sense to do so? 
- Steven prepared a new dataset for training GP model, which has the corrected traces (beta=0.5, 0.7, 0.9)

## Day 7 2025-11-18
- laptop drained and died this morning. all experiments were interrupted. the checkpoints were not saved properly. 
- Adam is looking to get us a desktop

## Day 8 2025-11-19
Notes during today's group meeting (Adam absent due to sickness)
- Steven's work on flowering detector is going well. His model can predict whether images are over 16 day to a high accuracy (> 95%). He also may try training a model that maps image embedding to area. He mentions that his new CV and image embedding works are in a different repo.
- Sam is tasked with making the 10min GP model (action = bright/low) work as well as possible, in two weeks time.
- I will continue to train the 1day GP, trying a few different things (different ways of packing action history, use the corrected dataset, etc). I will also train the InAC agent in the updated GP sim
- Oliver contributed helpful discussions and mentioned that he is currently swamped with other projects and will be freer after next jan