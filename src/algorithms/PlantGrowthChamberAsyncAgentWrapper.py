import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from algorithms.BaseAgent import BaseAgent
from algorithms.constants import BRIGHT_ACTION, TWILIGHT_INTENSITIES_30_MIN
from utils.RlGlue.agent import AsyncAgentWrapper

logger = logging.getLogger("PlantGrowthChamberAsyncAgentWrapper")
logger.setLevel(logging.DEBUG)


class PlantGrowthChamberAsyncAgentWrapper(AsyncAgentWrapper):
    def __init__(self, agent: BaseAgent):
        super().__init__(agent)
        self.action_timestep = timedelta(minutes=agent.params.get("action_timestep", 10))
        self.agent_started = False
        self.enforce_night = agent.params.get("enforce_night", True)
        self.last_action_time = None
        self.tz = ZoneInfo(agent.params.get("timezone", "Etc/UTC"))
        self.tz_utc = ZoneInfo("Etc/UTC")
        self.env_time = None
        self.env_local_time = None

    def update_time_from_extra(self, extra: dict[str, Any]) -> None:
        """Extract time from environment info dictionary."""
        self.env_time = datetime.fromtimestamp(extra["env_time"], tz=self.tz_utc)
        self.env_local_time = self.env_time.astimezone(self.tz)

    def is_dawn(self) -> bool:
        """Determine whether the environment time is within dawn hours (9:01 AM to 9:29 AM)."""
        assert self.env_local_time is not None, "Environment local time must be set before checking dawn."
        is_dawn = self.env_local_time.hour == 9 and 0 < self.env_local_time.minute < 30
        return is_dawn

    def is_dusk(self) -> bool:
        """Determine whether the environment time is within dusk hours (8:30 PM to 8:59 PM)."""
        assert self.env_local_time is not None, "Environment local time must be set before checking dusk."
        is_dusk = self.env_local_time.hour == 20 and 30 <= self.env_local_time.minute
        return is_dusk

    def is_night(self) -> bool:
        """Determine whether the environment time falls within nighttime hours."""
        assert self.env_local_time is not None, "Environment local time must be set before checking night."
        is_night = self.env_local_time.hour >= 21 or self.env_local_time.hour < 9 or (
            self.env_local_time.hour == 9 and self.env_local_time.minute < 1
        )
        return is_night

    def get_dawn_action(self) -> np.ndarray:
        """Calculate the appropriate light intensity for dawn based on current environment time."""
        assert self.env_local_time is not None, "Environment local time must be set before getting dawn action."
        minute_in_dawn = self.env_local_time.minute - 1

        if minute_in_dawn >= len(TWILIGHT_INTENSITIES_30_MIN):
            intensity = 1
        else:
            intensity = TWILIGHT_INTENSITIES_30_MIN[minute_in_dawn]

        logger.info(f"Dawn action at minute {minute_in_dawn}, intensity: {intensity}")
        return BRIGHT_ACTION * intensity

    def get_dusk_action(self) -> np.ndarray:
        """Calculate the appropriate light intensity for dusk based on current environment time."""
        if self.env_local_time is None:
            return np.zeros(6)

        minute_in_hour = self.env_local_time.minute

        # Calculate which twilight intensity to use
        idx = len(TWILIGHT_INTENSITIES_30_MIN) - (60 - minute_in_hour)

        # Safety check to ensure we're within array bounds
        if idx < 0:
            intensity = 1
        elif idx >= len(TWILIGHT_INTENSITIES_30_MIN):
            intensity = 0  # Complete darkness at end of dusk
        else:
            intensity = TWILIGHT_INTENSITIES_30_MIN[len(TWILIGHT_INTENSITIES_30_MIN) - 1 - idx]

        logger.info(f"Dusk action at minute {minute_in_hour}, intensity: {intensity}")
        return BRIGHT_ACTION * intensity

    def get_night_action(self) -> np.ndarray:
        """Return a zero action for night time (lights off)."""
        return np.zeros(6)

    async def start(self, observation: Any, extra: dict[str, Any] = {}) -> tuple[Any, dict[str, Any]]:
        self.update_time_from_extra(extra)

        if not self.maybe_enforce_action():
            logger.info(f"Starting agent at {self.env_local_time}")
            self.last_action_info = await asyncio.to_thread(self.agent.start, observation, extra)
            self.agent_started = True
        return self.last_action_info

    def maybe_enforce_action(self):
        if not self.enforce_night:
            return False
        if self.is_night():
            action = self.get_night_action()
            self.last_action_info = (action, {})
            logger.info(f"Enforcing night mode at {self.env_local_time}")
            return True
        if self.is_dawn():
            action = self.get_dawn_action()
            self.last_action_info = (action, {})
            logger.info(f"Enforcing dawn transition at {self.env_local_time}")
            return True
        if self.is_dusk():
            action = self.get_dusk_action()
            self.last_action_info = (action, {})
            logger.info(f"Enforcing dusk transition at {self.env_local_time}")
            return True
        return False

    async def step(self, reward: float, observation: Any, extra: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self.update_time_from_extra(extra)

        if self.maybe_enforce_action():
            return self.last_action_info

        if not self.agent_started:
            logger.info(f"Starting agent at {self.env_local_time}")
            self.last_action_info = await asyncio.to_thread(self.agent.start, observation, extra)
            self.agent_started = True
            return self.last_action_info

        # During daytime, poll the agent based on environment time
        # Only poll if enough time has passed since the last action
        assert self.env_time is not None, "Environment time must be set before checking action timestep."
        if self.last_action_time is None:
            self.last_action_time = self.env_time
        time_since_last_action = self.env_time - self.last_action_time

        action_timestep_minutes = self.action_timestep.total_seconds() / 60
        assert self.env_local_time is not None, "Environment local time must be set before checking action timestep."
        should_poll = self.env_local_time.minute % action_timestep_minutes == 0

        if time_since_last_action >= self.action_timestep or should_poll:
            logger.info(
                f"Polling agent at timestep mark: {self.env_local_time}, time since last action: {time_since_last_action}"
            )
            self.last_action_info = await asyncio.to_thread(self.agent.step, reward, observation, extra)
            self.last_action_time = self.env_time

        return self.last_action_info
