from typing import Any, Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class ConstantAgent(BaseAgent):
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
        self.action = self.params["constant_action"]

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.action, {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        return self.action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
