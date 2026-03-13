from typing import Any, Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class ContinuousRandomSimplexAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.low = params.get("intensity_min", 0.0)
        self.high = params.get("intensity_max", 1.0)
        self.color_alpha = params.get("color_alpha", 1.0)
        assert self.color_alpha > 0, "Parameter 'color_alpha' must be > 0."
        self.alpha = np.full(self.actions, self.color_alpha)

    def sample_action(self) -> np.ndarray:
        simplex = self.rng.dirichlet(self.alpha)
        intensity = self.rng.uniform(self.low, self.high)
        return simplex * intensity

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.sample_action(), {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        return self.sample_action(), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
