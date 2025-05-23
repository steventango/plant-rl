from typing import Optional
import gymnasium
from gymnasium.spaces import Box
from rlglue.environment import BaseEnvironment
import numpy as np

class Gym(BaseEnvironment):
    def __init__(self, name: str, seed: int, max_steps: Optional[int] = None):
        self.env = gymnasium.make(name, max_episode_steps=max_steps)
        self.seed = seed

        self.max_steps = max_steps

    def start(self):
        self.seed += 1
        s, info = self.env.reset(seed=self.seed)
        return np.asarray(s, dtype=np.float64).reshape(-1,), info

    def step(self, action):
        sp, r, terminated, truncated, info = self.env.step(action)

        return (np.asarray(sp, dtype=np.float64).reshape(-1,), float(r), terminated, truncated, info)
