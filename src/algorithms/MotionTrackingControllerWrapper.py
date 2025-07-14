import asyncio
import logging
from datetime import timedelta
from collections import defaultdict
from typing import Any, Dict
import numpy as np

from algorithms.BaseAgent import BaseAgent
from utils.RlGlue.agent import AsyncAgentWrapper
from utils.metrics import UnbiasedExponentialMovingAverage as uema
from utils.checkpoint import checkpointable

logger = logging.getLogger("MotionTrackingControllerWrapper")
logger.setLevel(logging.DEBUG)


@checkpointable(("sensitivity", "mean_areas", "openness_record", "openness_trace"))
class MotionTrackingControllerWrapper(AsyncAgentWrapper):
    def __init__(self, agent: BaseAgent):
        super().__init__(agent)
        self.env_local_time = None
        self.start_hour = 9
        self.end_hour = 21  # excluded in daytime
        self.time_step = 5  # minutes

        self.Imin = 0.5  # Lowest intensity. Fixed at a dim level at which CV still functions well.
        self.Imax = 1.0  # Highest intensity. Its optimal value depends on plant species, developmental stage, and environmental factors. Tuned by the RL agent daily.
        self.sensitivity = 8.0  # = (change in intensity) / (change in plants openness). Adjusted daily by motion tracker to attempt to reach Imax at max openness.

        self.mean_areas = defaultdict(float)
        self.openness_record = []
        self.openness_trace = uema(alpha=0.1)

        # RL agent stuff
        self.agent_started = False
        self.Imax_lowerbound = 0.7
        self.Imax_upperbound = 1.3
        self.Imax_increment = 0.1

    def get_action(self) -> float:
        openness = self.openness_trace.compute().item()
        return max(min(self.Imin + self.sensitivity * openness, self.Imax), self.Imin)

    def adjust_sensitivity(self):
        max_openness = np.mean(np.sort(self.openness_record)[-5:])
        self.sensitivity = (self.Imax - self.Imin) / max_openness
        logger.info(
            f"New Imax = {self.Imax}. New sensitivity = {self.sensitivity:.2f}."
        )

    def reward(self):
        current_time = self.env_local_time.replace(second=0, microsecond=0)
        current_area = self.mean_areas.get(current_time, -1)
        yesterday_area = self.mean_areas.get(current_time - timedelta(days=1), -1)
        return current_area / yesterday_area - 1

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

    async def step(self, reward: float, observation: np.ndarray, extra: Dict[str, Any]):
        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_areas[self.env_local_time.replace(second=0, microsecond=0)] = float(
            mean_area
        )

        # Poll RL agent every morning (starting on day 2)
        if self.is_first_tod and self.openness_record != []:
            if not self.agent_started:
                logger.info(f"Starting RL agent at {self.env_local_time}")
                action = await asyncio.to_thread(
                    self.agent.start,
                    np.hstack([self.openness_record, self.Imax]),
                    extra,
                )
                tune_Imax = action * self.Imax_increment
                self.agent_started = True
            else:
                logger.info(f"Polling RL agent at {self.env_local_time}.")
                action = await asyncio.to_thread(
                    self.agent.step,
                    self.reward(),
                    np.hstack([self.openness_record, self.Imax]),
                    extra,
                )
                tune_Imax = action * self.Imax_increment

            self.Imax = min(
                max(self.Imax + tune_Imax, self.Imax_lowerbound), self.Imax_upperbound
            )
            self.adjust_sensitivity()

            self.openness_record = []
            self.openness_trace.reset()

        if self.is_night():
            action = 0.0
        elif self.is_zeroth_tod():
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

    def is_first_tod(self) -> bool:
        assert self.env_local_time is not None, (
            "Environment local time must be set before checking is_first_tod."
        )
        is_first_tod = (
            self.env_local_time.hour == self.start_hour
            and self.env_local_time.minute == self.time_step
        )
        return is_first_tod


class MotionTrackingControllerWrapper_NoTracking(MotionTrackingControllerWrapper):
    def get_action(self) -> float:
        return self.Imax
