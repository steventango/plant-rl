import numpy as np
from datetime import datetime, date
import logging
from utils.RlGlue.agent import BaseAsyncAgent
from utils.constants import BALANCED_ACTION_100
from utils.checkpoint import checkpointable

logger = logging.getLogger("plant_rl.ScheduleAgent")


@checkpointable(())
class ScheduleAgent(BaseAsyncAgent):
    def __init__(self, observations, actions, params, collector=None, seed=None):
        self._init_args = (observations, actions, params, collector, seed)
        self.params = params
        self.action_days = params.get("action_days", [1])
        self.action_inputs = params.get("action_inputs", [0.0])
        self.start_hour = 9
        self.end_hour = 21
        self.start_date = date.fromisoformat(params.get("local_start_date"))

    def _is_night(self, t: datetime) -> bool:
        return t.hour >= self.end_hour or t.hour < self.start_hour

    def _is_photo_time(self, t: datetime) -> bool:
        return t.hour == self.start_hour - 1 and t.minute == 59

    def _get_scalar_action(self, t: datetime) -> float:
        day_number = (t.date() - self.start_date).days + 1
        scalar_action = float(self.action_inputs[0])
        for day, value in zip(self.action_days, self.action_inputs, strict=False):
            if day_number >= day:
                scalar_action = float(value)
            else:
                break
        return scalar_action

    def _get_action(self, t: datetime) -> float | np.ndarray:
        if self._is_photo_time(t):
            return 0.4 * BALANCED_ACTION_100  # 40 ppfd of flash for camera image
        elif self._is_night(t):
            return np.zeros(6)
        return self._get_scalar_action(t)

    async def start(self, observation, extra=None):
        t, _ = observation
        logger.debug(f"Today is Day {(t.date() - self.start_date).days + 1}.")
        return self._get_action(t), {}

    async def step(self, reward: float, observation, extra):
        t, _ = observation
        return self._get_action(t), {}

    async def end(self, reward: float, extra):
        return {}

    async def plan(self) -> None:
        pass
