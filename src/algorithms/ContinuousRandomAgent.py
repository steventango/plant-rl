from typing import Any, Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class ContinuousRandomAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        self.updates = 0
        self.low = params.get("action_min", 0)
        self.high = params.get("action_max", 1)
        self._action_size = None if self.actions == 1 else self.actions

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.rng.uniform(self.low, self.high, self._action_size), {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        return self.rng.uniform(self.low, self.high, self._action_size), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
