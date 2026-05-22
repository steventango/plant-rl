import numpy as np
from datetime import datetime
from utils.RlGlue.agent import BaseAsyncAgent
from utils.constants import BALANCED_ACTION_100
from utils.checkpoint import checkpointable


@checkpointable(("start_date",))
class ScheduleAgent(BaseAsyncAgent):
    def __init__(self, observations, actions, params, seed):
        self._init_args = (observations, actions, params, seed)
        self.params = params
        self.action_days = params.get("action_days", [1])
        self.action_inputs = params.get("action_inputs", [0.0])
        self.start_hour = 9
        self.end_hour = 21
        self.start_date = None

    def _is_night(self, t: datetime) -> bool:
        return t.hour >= self.end_hour or t.hour < self.start_hour

    def _is_photo_time(self, t: datetime) -> bool:
        return t.hour == self.start_hour - 1 and t.minute == 59

    def _get_scalar_action(self, t: datetime) -> float:
        if self.start_date is None:
            return float(self.action_inputs[0])
        day_number = (t.date() - self.start_date).days + 1
        scalar_action = float(self.action_inputs[0])
        for day, value in zip(self.action_days, self.action_inputs, strict=False):
            if day_number >= day:
                scalar_action = float(value)
            else:
                break
        return scalar_action

    def _get_action(self, t: datetime) -> float | np.ndarray:
        if self._is_night(t):
            return np.zeros(6)
        if self._is_photo_time(t):
            return 0.4 * BALANCED_ACTION_100
        return self._get_scalar_action(t)

    async def start(self, observation, extra=None):
        t, _ = observation
        self.start_date = t.date()
        return self._get_action(t), {}

    async def step(self, reward: float, observation, extra):
        t, _ = observation
        return self._get_action(t), {}

    async def end(self, reward: float, extra):
        return {}
