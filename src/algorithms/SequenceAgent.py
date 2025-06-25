import json
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class SequenceAgent(BaseAgent):
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
        self.actions = [
            np.array(action) for action in json.loads(self.params["actions"])
        ]

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = self.actions[self.steps]
        self.steps += 1
        return action, {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        action = self.actions[self.steps]
        self.steps += 1
        return action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
