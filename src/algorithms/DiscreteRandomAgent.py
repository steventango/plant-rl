from typing import Any, Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.policies import sample


class DiscreteRandomAgent(BaseAgent):
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
        self.pi = np.full(self.actions, 1 / self.actions)

    def sample_action(self) -> int:
        """
        Sample an action uniformly at random from the available actions.
        """
        return sample(self.pi, rng=self.rng)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.sample_action(), {}  # type: ignore

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        return self.sample_action(), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
