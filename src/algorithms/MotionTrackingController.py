import logging  # type: ignore
from collections import defaultdict
from math import tanh
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.metrics import UnbiasedExponentialMovingAverage as uema

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
        self.start_hour = 13  # included in daytime
        self.end_hour = 21  # excluded in daytime

        self.env_local_time = None
        self.mean_clean_areas = defaultdict(float)
        self.openness_trace = uema(alpha=0.1)

        self.Imin = 0.35  # lowest allowable intensity during daytime. Fixed at "moonlight" level.
        self.Imax = 1.0  # highest allowable intensity. Can be tuned by higher-level RL
        self.sensitivity = 5.0  # roughly (change in intensity) / (change in plants openness). Can be tuned by higher-level RL

    def is_night(self) -> bool:
        assert self.env_local_time is not None, (
            "Environment local time must be set before checking night."
        )
        is_night = (
            self.env_local_time.hour >= self.end_hour
            or self.env_local_time.hour < self.start_hour
        )
        return is_night

    def is_zeroth_tod(self) -> bool:
        assert self.env_local_time is not None, (
            "Environment local time must be set before checking is_zeroth_tod."
        )
        is_zeroth_tod = (
            self.env_local_time.hour == self.start_hour
            and self.env_local_time.minute == 0
        )
        return is_zeroth_tod

    def get_action(self) -> float:
        openness = self.openness_trace.compute().item()
        return self.Imin + max(
            0, (self.Imax - self.Imin) * tanh(self.sensitivity * openness)
        )

    def start(self, observation: np.ndarray, extra: Dict[str, Any]):  # type: ignore
        self.openness_trace.reset()

        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_clean_areas[self.env_local_time.replace(second=0, microsecond=0)] = (
            float(mean_area)
        )

        if self.is_night():
            action = 0.0
        elif self.is_zeroth_tod():
            action = self.Imin
        else:
            logger.warning(
                "Starting motion-tracking controller during daytime. Enforce standard lighting until tomorrow."
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
            action = 0.0
        elif self.is_zeroth_tod():
            self.openness_trace.reset()
            action = self.Imin
        else:
            today_zeroth_time = self.env_local_time.replace(
                hour=self.start_hour, minute=0, second=0, microsecond=0
            )
            today_first_time = self.env_local_time.replace(
                hour=self.start_hour, minute=1, second=0, microsecond=0
            )
            today_zeroth_area = self.mean_clean_areas.get(today_zeroth_time, -1)
            today_first_area = self.mean_clean_areas.get(today_first_time, -1)
            if today_zeroth_area == -1 or today_first_area == -1:
                logger.warning(
                    f"No same-day morning measurement available at {self.env_local_time}. Enforce standard lighting."
                )
                action = 1.0
            elif today_first_area == 0.0:   
                logger.warning(
                    f"Same-day morning measurement at {self.env_local_time} is zero. Enforce standard lighting."
                )
                action = 1.0
            else:
                self.openness_trace.update(mean_area / today_first_area - 1)
                action = self.get_action()

        return action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
