from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.policies import sample


class DiscreteRandomAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        self.updates = 0
        self.pi = np.full(self.actions, 1 / self.actions)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, observation: np.ndarray):
        return sample(self.pi, rng=self.rng), {}

    def step(self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]):
        return sample(self.pi, rng=self.rng), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        pass
