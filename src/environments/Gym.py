from typing import Optional  # type: ignore

import gymnasium
import numpy as np
from RlGlue.environment import BaseEnvironment


class Gym(BaseEnvironment):
    def __init__(self, name: str, seed: int, max_steps: Optional[int] = None):
        self.env = gymnasium.make(name, max_episode_steps=max_steps)
        self.seed = seed

        self.max_steps = max_steps

    def start(self):
        self.seed += 1
        s, info = self.env.reset(seed=self.seed)
        return np.asarray(s, dtype=np.float64).reshape(
            -1,
        ), info

    def step(self, a):  # type: ignore
        sp, r, t, _, info = self.env.step(a)

        return (
            r,
            np.asarray(sp, dtype=np.float64).reshape(
                -1,
            ),
            t,
            {},
        )
