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

    def get_time(self):
        return datetime.now(tz=self.tz_utc)

    def get_local_time(self):
        return self.get_time().astimezone(self.tz)

    def is_dawn(self) -> bool:
        """
        Determine whether the current local time is within dawn hours (9:00 AM to 9:29 AM).
        """
        local_time = self.get_local_time()
        is_dawn = local_time.hour == 9 and local_time.minute < len(TWILIGHT_INTENSITIES_30_MIN)
        return is_dawn

    def is_dusk(self) -> bool:
        """
        Determine whether the current local time is within dusk hours (8:30 PM to 8:59 PM).
        """
        local_time = self.get_local_time()
        is_dusk = local_time.hour == 20 and local_time.minute >= (60 - len(TWILIGHT_INTENSITIES_30_MIN))
        return is_dusk

    def is_night(self) -> bool:
        """
        Determine whether the given time falls within nighttime hours.
        """
        local_time = self.get_local_time()
        is_night = local_time.hour >= 21 or local_time.hour < 9
        return is_night

    def get_dawn_action(self) -> np.ndarray:
        """Calculate the appropriate light intensity for dawn based on current time."""
        current_local_time = self.get_local_time()
        minute_in_dawn = current_local_time.minute

        if minute_in_dawn >= len(TWILIGHT_INTENSITIES_30_MIN):
            intensity = TWILIGHT_INTENSITIES_30_MIN[-1]
        else:
            intensity = TWILIGHT_INTENSITIES_30_MIN[minute_in_dawn]

        logger.info(f"Dawn action at minute {minute_in_dawn}, intensity: {intensity}")
        return BRIGHT_ACTION * intensity

    def get_dusk_action(self) -> np.ndarray:
        """Calculate the appropriate light intensity for dusk based on current time."""
        current_local_time = self.get_local_time()
        minute_in_hour = current_local_time.minute

        # Calculate which twilight intensity to use
        idx = len(TWILIGHT_INTENSITIES_30_MIN) - (60 - minute_in_hour)

        # Safety check to ensure we're within array bounds
        if idx < 0:
            intensity = TWILIGHT_INTENSITIES_30_MIN[0]
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
        night_enforced = self.maybe_enforce_night()
        if not night_enforced:
            self.last_action_info = await asyncio.to_thread(self.agent.start, observation, extra)
            self.agent_started = True
        return self.last_action_info

    def maybe_enforce_night(self):
        if not self.enforce_night:
            return False
        if self.is_night():
            action = self.get_night_action()
            self.last_action_info = (action, {"agent_info": {}})
            return True
        if self.is_dawn():
            action = self.get_dawn_action()
            self.last_action_info = (action, {"agent_info": {}})
            return True
        if self.is_dusk():
            action = self.get_dusk_action()
            self.last_action_info = (action, {"agent_info": {}})
            return True

    async def step(self, reward: float, observation: Any, extra: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        current_time = self.get_local_time()

        night_enforced = self.maybe_enforce_night()
        if night_enforced:
            return self.last_action_info

        if not self.agent_started:
            self.last_action_info = await asyncio.to_thread(self.agent.start, observation, extra)
            return self.last_action_info

        # During daytime, poll the agent every action_timestep minutes
        now = datetime.now()
        action_timestep = self.action_timestep
        time_since_last_action = now - (self.last_action_time if self.last_action_time else now)
        action_timestep_minutes = action_timestep.total_seconds() / 60
        should_poll = now.minute % action_timestep_minutes == 0

        if time_since_last_action > action_timestep or should_poll:
            logger.info(
                f"Polling agent at timestep mark: {current_time}, time since last action: {time_since_last_action}"
            )
            self.last_action_info = await asyncio.to_thread(self.agent.step, reward, observation, extra)
            self.last_action_time = now

        return self.last_action_info
