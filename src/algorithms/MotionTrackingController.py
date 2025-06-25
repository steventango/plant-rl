from typing import Any, Dict, Tuple
import logging

import numpy as np
from math import tanh
from collections import defaultdict
from utils.metrics import UnbiasedExponentialMovingAverage as uema

from PyExpUtils.collection.Collector import Collector
from algorithms.BaseAgent import BaseAgent

logger = logging.getLogger("MotionTrackingController")
logger.setLevel(logging.DEBUG)


class MotionTrackingController(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.env_local_time = None
        self.mean_clean_areas = defaultdict(float)
        self.openness_trace = uema(alpha=0.1)

        # Tunable parameters of motion-tracking controller
        self.moonlight = 0.35
        self.c1 = (
            1.1 - self.moonlight
        )  # determines max intensity, which is self.moonlight + self.c1
        self.c2 = 10.0  # determines how fast to ramp up the intensity as area changes

    def is_night(self) -> bool:
        assert self.env_local_time is not None, (
            "Environment local time must be set before checking night."
        )
        is_night = self.env_local_time.hour >= 21 or self.env_local_time.hour < 9
        return is_night

    def get_action(self) -> float:
        openness = self.openness_trace.compute().item()
        return self.moonlight + max(0, self.c1 * tanh(self.c2 * openness))

    def start(self, observation: np.ndarray, extra: Dict[str, Any]):
        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_clean_areas[self.env_local_time.replace(second=0, microsecond=0)] = (
            float(mean_area)
        )

        if self.is_night():
            action = 0.0
        elif self.env_local_time.hour == 9 and self.env_local_time.minute == 0:
            action = self.get_action()
        else:
            logger.warning(
                "Starting motion-tracking controller in the middle of the day. Enforce standard lighting until tomorrow."
            )
            action = 1.0

        return action, {}

    def step(self, reward: float, observation: np.ndarray, extra: Dict[str, Any]):
        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_clean_areas[self.env_local_time.replace(second=0, microsecond=0)] = (
            float(mean_area)
        )

        if self.is_night():
            self.openness_trace.reset()
            action = 0.0
        else:
            today_morning_time = self.env_local_time.replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            today_morning_area = self.mean_clean_areas.get(today_morning_time, 0.0)
            if today_morning_area == 0:
                logger.warning(
                    f"No same-day morning measurement available at {self.env_local_time}. Enforce standard lighting."
                )
                action = 1.0
            else:
                self.openness_trace.update(mean_area / today_morning_area - 1)
                action = self.get_action()

        return action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
