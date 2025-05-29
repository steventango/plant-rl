#  Experiment E5 -- Offline learning from Exp 3 data
##  Phase P0
### Objectives: 
- Perform offline learning using data from Exp 3
- Test the following design: state = (TOC, daily photon counts), reward = daily growth
### Methods: 
- Set up the code base for offline learning 
- Let reward be the difference between daily peaks, given at the last time stamp of each day
- Daily photon counts is a way to keep track of how much light the plants have received so far today. In the simple action space [0, 1], let photon_counts = sum(actions taken so far today)
- Everything is scaled to its daily values. 
### Observations: 
- If using only constant-agent's data, ![alt text](image-3.png)
- If using all data, ![alt text](image-1.png)
- If using only non-constant-agents' data, ![alt text](image-2.png)
- These imply that our data is not good for offline learning. 
### Conclusions & Outlooks: 
- How to better collect data?

##  Phase P0
### Objectives: 
- Since our Exp 3 data was collected at 10 min time step & and we want TOD to be every hour, a state often goes back to itself. I want to avoid this by simulating 1hr time step using Exp 3 data.
### Methods: 
- See env subclass OfflinePlantGrowthChamber_1hrStep
- Use only non-constant agents' data
- Replay 500000 steps with batch size 64
### Observations: 
- Advantage and policy:
![alt text](image.png)
### Conclusions & Outlooks: 
- 