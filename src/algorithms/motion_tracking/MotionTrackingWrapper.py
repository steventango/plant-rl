import asyncio
import logging
from datetime import timedelta
from collections import defaultdict
from typing import Any, Dict
import numpy as np
#import matplotlib.pyplot as plt

from algorithms.BaseAgent import BaseAgent
from utils.RlGlue.agent import AsyncAgentWrapper
from utils.metrics import UnbiasedExponentialMovingAverage as uema
# from utils.checkpoint import checkpointable

logger = logging.getLogger("plant_rl.MotionTrackingWrapper")
logger.setLevel(logging.DEBUG)


# @checkpointable(("sensitivity", "mean_areas", "openness_record", "openness_trace"))
class MotionTrackingWrapper(AsyncAgentWrapper):
    def __init__(self, agent: BaseAgent):
        super().__init__(agent)
        # Wrapper params
        self.start_hour = 9
        self.end_hour = 21  # excluded in daytime
        self.time_step = 5  # minutes
        self.morning_intensity = (
            50.0  # Fixed at a dim level at which CV still functions well
        )
        self.target_intensity = 100.0  # Its optimal value depends on plant species, developmental stage, and environmental factors. Tuned by the RL agent daily.

        # Agent params
        self.agent_Imax = 150.0
        self.agent_increment = 50
        self.actions = [-1, 0, 1]

        self.is_first_day = True
        self.agent_started = False

        self.sensitivity = uema(
            alpha=0.1
        )  # = (change in intensity) / (change in plants openness). Adjusted daily by motion tracker to attempt to reach target_intensity at max openness.
        self.mean_areas = defaultdict(float)
        self.openness_record = []
        self.openness_trace = uema(alpha=0.1)

    def get_action(self) -> float:
        if self.is_first_day:
            return self.target_intensity
        else:
            openness = self.openness_trace.compute().item()
            if openness < 0:
                return self.morning_intensity
            else:
                proposed_action = (
                    self.morning_intensity
                    + self.sensitivity.compute().item() * np.log(100 * openness + 1)
                )
                return min(proposed_action, self.agent_Imax)

    def adjust_sensitivity(self):
        max_openness = np.mean(np.sort(self.openness_record)[-5:])
        proposed_sensitivity = (
            self.target_intensity - self.morning_intensity
        ) / np.log(100 * max_openness + 1)
        self.sensitivity.update(proposed_sensitivity)
        logger.info(
            f"Set target_intensity = {self.target_intensity}"
        )  # Set sensitivity = {self.sensitivity.compute().item():.2f}."

    def reward(self):
        current_time = self.env_local_time.replace(second=0, microsecond=0)
        current_area = self.mean_areas.get(current_time, -1)
        yesterday_area = self.mean_areas.get(current_time - timedelta(days=1), -1)
        if yesterday_area == -1 or current_area == -1:
            return 0.0
        else:
            return 100 * (current_area / yesterday_area - 1)

    async def start(self, observation: np.ndarray, extra: Dict[str, Any]):  # type: ignore
        self.agent_started = False
        self.is_first_day = True
        self.mean_areas = defaultdict(float)
        self.openness_trace.reset()
        self.openness_record = []
        self.sensitivity.reset()

        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_areas[self.env_local_time.replace(second=0, microsecond=0)] = float(
            mean_area
        )

        if self.is_night():
            action = 0.0
        elif self.is_zeroth_tod():
            action = self.morning_intensity
        else:
            logger.warning(
                f"Starting motion-tracking controller during daytime. Enforce target_intensity = {self.target_intensity} until tomorrow."
            )
            action = self.target_intensity

        return action, {}

    async def step(self, reward: float, observation: np.ndarray, extra: Dict[str, Any]):
        # Get observations
        self.env_local_time = observation[0]
        mean_area = observation[1]
        self.mean_areas[self.env_local_time.replace(second=0, microsecond=0)] = float(
            mean_area
        )

        # Poll RL agent every morning
        if self.is_first_tod():
            await self.poll_agent(extra)

        # Return next intensity in ppfd
        if self.is_night():
            action = 0.0
        elif self.is_zeroth_tod():
            action = self.morning_intensity
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
                    f"No same-day morning measurement available at {self.env_local_time}. Enforce target_intensity = {self.target_intensity}."
                )
                action = self.target_intensity
            elif today_first_area == 0.0:
                logger.warning(
                    f"Same-day morning measurement at {self.env_local_time} is zero. Enforce target_intensity = {self.target_intensity}."
                )
                action = self.target_intensity
            else:
                openness = mean_area / today_first_area - 1
                self.openness_trace.update(openness)
                self.openness_record.append(openness)
                action = self.get_action()

        return action, {}

    async def end(self, reward: float, extra: Dict[str, Any]):
        return {}

    async def poll_agent(self, extra):
        # if just starting new run, the previous day was the terminal day
        if (
            len(self.openness_record)
            != (self.end_hour - self.start_hour) * 60 / self.time_step - 1
        ):
            _ = await asyncio.to_thread(
                self.agent.end,
                0.0,
                extra,
            )
            return

        # Use daily openness curve as state
        ob = np.array(self.openness_record)
        ob = ob / np.max(ob)

        if not self.agent_started:
            # logger.info(f"Starting RL agent at {self.env_local_time}")
            agent_action, _ = await asyncio.to_thread(
                self.agent.start,
                ob,
                extra,
            )
            change_target_intensity = self.actions[agent_action] * self.agent_increment
            self.agent_started = True
            self.is_first_day = False
        else:
            # logger.info(f"Polling RL agent at {self.env_local_time}.")
            agent_action, _ = await asyncio.to_thread(
                self.agent.step,
                self.reward(),
                ob,
                extra,
            )

            change_target_intensity = self.actions[agent_action] * self.agent_increment

        self.target_intensity = min(
            max(
                self.target_intensity + change_target_intensity, self.morning_intensity
            ),
            self.agent_Imax,
        )

        self.adjust_sensitivity()

        self.openness_record = []
        self.openness_trace.reset()

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


class MotionTrackingWrapper_NoTracking(MotionTrackingWrapper):
    def get_action(self) -> float:
        return self.target_intensity
