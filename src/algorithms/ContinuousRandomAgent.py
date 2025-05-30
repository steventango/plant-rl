from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class ContinuousRandomAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        self.updates = 0
        self.low = 0
        self.high = 1

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, observation: np.ndarray):
        return self.rng.uniform(self.low, self.high, self.actions), {}

    def step(self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]):
        return self.rng.uniform(self.low, self.high, self.actions), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
