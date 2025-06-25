from typing import Any, Dict, Tuple

import numpy as np
from math import tanh
from utils.metrics import UnbiasedExponentialMovingAverage as uema

from PyExpUtils.collection.Collector import Collector
from algorithms.BaseAgent import BaseAgent

class MotionTrackingRecipe(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.moonlight = 0.35
        self.openness_trace = uema(alpha=0.1)

        # Tunable parameters of the light recipe
        self.c1 = 1.0 - self.moonlight     # determines max intensity, which is self.moonlight + self.c1
        self.c2 = 10.0                     # determines how fast to ramp up the intensity as area changes

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, observation: np.ndarray, extra: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        local_time = observation[0]
        mean_clean_area = observation[1]

        self.previous_date = local_time.date()

        self.openness_trace.reset()
        self.morning_area = np.copy(mean_clean_area)
        self.openness_trace.update(0.0)

        return self.moonlight, {}

    def step(self, reward: float, observation: np.ndarray, extra: Dict[str, Any]):
        local_time = observation[0]
        mean_clean_area = observation[1]

        if local_time.date != self.previous_date:
            self.openness_trace.reset()
            self.openness_trace.update(0.0)
            self.morning_area = np.copy(mean_clean_area)
            self.previous_date = local_time.date()
        else:
            self.openness_trace.update(mean_clean_area / self.morning_area - 1)

        openness = self.openness_trace.compute().item()

        return self.moonlight + max(0, self.c1*tanh(self.c2*openness)), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
