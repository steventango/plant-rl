import random 
import numpy as np
from environments.PlantSimulator import PlantSimulator

rand_state = np.random.RandomState(0)
env = PlantSimulator(random_plant=True, n_step=1)
env.start()
for i in range(1000):
    r, s, t, _ = env.step(rand_state.randint(0, 1))
    if t:
        env.start()