from typing import Optional
import gymnasium
from gymnasium.spaces import Box
from rlglue.environment import BaseEnvironment
import numpy as np

class CliffWalking(BaseEnvironment):
    def __init__(self, max_steps: Optional[int] = None):
        self.env = gymnasium.make('CliffWalking-v0')

    def start(self):
        s, info = self.env.reset()
        return self.one_hot_state(s)

    def step(self, action):
        sp, r, terminated_gym, truncated_gym, info = self.env.step(action) # gymnasium returns terminated and truncated
        
        # If the underlying gym env only returned a single 'done' flag (as 't' in the old code):
        # terminated = t
        # truncated = False

        return self.one_hot_state(sp), float(r), terminated_gym, truncated_gym, {}
    
    def one_hot_state(self, state):
        one_hot = np.zeros(self.env.observation_space.n, dtype=np.float64)
        one_hot[state] = 1
        return one_hot