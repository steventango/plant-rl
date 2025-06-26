from typing import Optional  # type: ignore

import gymnasium
import numpy as np
from RlGlue.environment import BaseEnvironment


class CliffWalking(BaseEnvironment):
    def __init__(self, max_steps: Optional[int] = None):
        self.env = gymnasium.make("CliffWalking-v0")

    def start(self):
        s, info = self.env.reset()
        return self.one_hot_state(s)

    def step(self, a):  # type: ignore
        sp, r, t, _, info = self.env.step(a)

        return (r, self.one_hot_state(sp), t, {})

    def one_hot_state(self, state):
        one_hot = np.zeros(self.env.observation_space.n, dtype=np.float64)  # type: ignore
        one_hot[state] = 1
        return one_hot
