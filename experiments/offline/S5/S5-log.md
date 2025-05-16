#  Experiment S5 -- Sample efficient learning
##  Phase P0

### Objectives:
See if replay ratio can improve sample efficiency.

### Methods:
Environment:
- Reward: change in plant area normalized by the first area of the day
Agents:
- ESARSA_replay
  - Fork of E4/S8 as baseline
  - ESARSA(0)
  - Replay
  - Step size decay from 0.1 to 0.01 in 5e4 steps
  - Batch size 32
- ESARSA_replay_no_decay
  - ESARSA(0)
  - Replay
  - Fixed step size 0.03
  - Batch size 32
- ESARSA_replay_no_decay_big_batch
  - ESARSA(0)
  - Replay
  - Step size 0.03
  - Batch size 256
- ESARSA_replay_ratio_16
  - ESARSA(0)
  - Replay
  - Step size 0.01
  - Batch size 256
  - Replay ratio 16

### Observations:
- No step size decay actually reaches optimal policy faster than step size decay
- Bigger batch size did not improve sample efficiency, but it improved the final performance
- Bigger replay ratio can improve sample efficiency
### Conclusions & Outlooks:
- Step size decay is not necessary
- Replay ratio is promising


##  Phase P1

### Objectives:
Sweep over step size, batch size, and replay ratio to see what leads to the most
sample efficient agent.

### Methods:
Environment:
- Reward: change in plant area normalized by the first area of the day
Agents:
- ESARSA_replay / ESARSA_replay_ratio
  - ESARSA(0)
  - Replay
  - Step size: [0.03, 0.01, 0.003, 0.001, 0.0003]
  - Batch size: [16, 32, 64, 128, 256]
  - Replay ratio: [1, 4, 8, 16, 32, 64, 128]

### Observations:
- ?
### Conclusions & Outlooks:
- ?
