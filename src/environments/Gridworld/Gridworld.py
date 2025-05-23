import gymnasium
from rlglue.environment import BaseEnvironment
import numpy as np

class Gridworld(BaseEnvironment):
    def __init__(self):
        self.env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

    def start(self):
        s, info = self.env.reset()
        return self.one_hot_state(s)

    def step(self, action):
        sp, r, terminated, truncated_gym, info = self.env.step(action) # gymnasium returns terminated and truncated
        # Assuming truncated_gym is the truncated flag. If not, and only a single 'done' is available,
        # then terminated = t, truncated = False.
        # If the underlying env (gymnasium) gives both, we use them.
        
        # If the gym env only returns a single done flag (often called `t` or `terminated`):
        # sp, r, t, info = self.env.step(action) # if it were a 4-tuple return
        # terminated = t
        # truncated = False

        return self.one_hot_state(sp), float(r), terminated, truncated_gym, {}
    
    def one_hot_state(self, state):
        one_hot = np.zeros(self.env.observation_space.n, dtype=np.float64)
        one_hot[state] = 1
        return one_hot