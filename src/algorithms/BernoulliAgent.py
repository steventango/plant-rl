from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.policies import sample


class BernoulliAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        self.updates = 0
        self.p = params.get('p', 0.5)
        assert actions == 2, "BernoulliAgent only supports two actions (0 and 1)."
    
    def sample_action(self):
        return np.random.binomial(1, self.p)
    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, observation: np.ndarray):
        return self.sample_action(), {}

    def step(self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]):
        return self.sample_action(), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
