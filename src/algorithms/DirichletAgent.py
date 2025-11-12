from typing import Any, Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION


class DirichletAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector | None,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        # Dirichlet parameters, all 1 for uniform
        self.alpha = np.ones(3)
        self.project_action = params.get("project_action", True)

    def sample_action(self) -> np.ndarray:
        # Sample from Dirichlet(1,1,1)
        z = self.rng.dirichlet(self.alpha)
        # Compute action: RED*z[0] + WHITE*z[1] + BLUE*z[2]
        if not self.project_action:
            return z
        action = z[0] * RED_ACTION + z[1] * BALANCED_ACTION_105 + z[2] * BLUE_ACTION
        return action

    # ----------------------
    # -- RLGlue interface --
    # ----------------------

    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = self.sample_action()
        return action, {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        action = self.sample_action()
        return action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
