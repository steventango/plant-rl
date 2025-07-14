import logging  # type: ignore
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.metrics import UnbiasedExponentialMovingAverage as uema
from utils.checkpoint import checkpointable

logger = logging.getLogger("plant_rl.MotionTrackingController")
logger.setLevel(logging.DEBUG)


# @checkpointable(("w", "theta"))
@checkpointable(
    (
        "sensitivity",
        "morning_area",
        "openness_record",
        "openness_trace",
        "env_local_time",
    )
)
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
        self.start_hour = 9
        self.end_hour = 21  # excluded in daytime
        self.time_step = 5  # minutes
        self.steps_per_day = int(
            (self.end_hour - self.start_hour) * 60 / self.time_step
        )

        self.env_local_time = None
        self.mean_areas = defaultdict(float)
        self.openness_trace = uema(alpha=0.1)  # need to be a scalar
        self.openness_record = []
        self.morning_area = None

        self.Imin = 0.5  # Lowest intensity. Fixed at a dim level at which CV still functions well.
        self.Imax = 1.1  # Highest intensity. Its optimal value depends on plant species, developmental stage, and environmental factors. Can be tuned by a higher-level RL agent
        self.sensitivity = 5.0  # = (change in intensity) / (change in plants openness). Adjusted daily to attempt to reach Imax when openness is the largest.

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
        return max(min(self.Imin + self.sensitivity * openness, self.Imax), self.Imin)

    def adjust_sensitivity(self):
        if self.openness_record != []:
            max_openness = np.mean(np.sort(self.openness_record)[-5:])
            self.sensitivity = (self.Imax - self.Imin) / max_openness
            logger.debug(f"Adjusted sensitivity = {self.sensitivity:.2f}")

    def start(self, observation: np.ndarray, extra: Dict[str, Any]):  # type: ignore
        self.openness_trace.reset()
        self.openness_record = []

        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_areas[self.env_local_time.replace(second=0, microsecond=0)] = float(
            mean_area
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
        self.mean_areas[self.env_local_time.replace(second=0, microsecond=0)] = float(
            mean_area
        )

        if self.is_night():
            action = 0.0
        elif self.is_zeroth_tod():
            self.adjust_sensitivity()
            self.openness_record = []
            self.openness_trace.reset()
            action = self.Imin
        else:
            today_zeroth_time = self.env_local_time.replace(
                hour=self.start_hour, minute=0, second=0, microsecond=0
            )
            today_first_time = self.env_local_time.replace(
                hour=self.start_hour,
                minute=self.time_step,
                second=0,
                microsecond=0,
            )
            today_zeroth_area = self.mean_areas.get(today_zeroth_time, -1)
            today_first_area = self.mean_areas.get(today_first_time, -1)
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
                openness = mean_area / today_first_area - 1
                self.openness_trace.update(openness)
                self.openness_record.append(openness)
                action = self.get_action()

        return action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
