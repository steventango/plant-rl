import random 
import numpy as np
from environments.PlantSimulator import PlantSimulator

rand_state = np.random.RandomState(0)
for i in range(100):
    env = PlantSimulator(random_plant=True, n_step=1, seed=random.randint(0, 1000))
    env.start()
    for i in range(2000):
        r, s, t, _ = env.step(random.randint(0, 1))
        if t:
            env.start()