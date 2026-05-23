import numpy as np
from datetime import datetime
from utils.RlGlue.agent import BaseAsyncAgent
from utils.constants import BALANCED_ACTION_100
from utils.checkpoint import checkpointable


@checkpointable(())
class Incubator(BaseAsyncAgent):
    def __init__(self, observations, actions, params, seed):
        self._init_args = (observations, actions, params, seed)
        self.params = params
        agent_params = params.get("agent", {})
        self.incubation_ppfd = float(agent_params.get("incubation_ppfd", 100.0))
        self.start_hour = 9
        self.end_hour = 21

    def _is_night(self, t: datetime) -> bool:
        return t.hour >= self.end_hour or t.hour < self.start_hour

    def _is_photo_time(self, t: datetime) -> bool:
        return t.hour == self.start_hour - 1 and t.minute == 59

    def _get_action(self, t: datetime) -> np.ndarray:
        if self._is_photo_time(t):
            return 0.4 * BALANCED_ACTION_100
        elif self._is_night(t):
            return np.zeros(6)
        return self.incubation_ppfd / 100 * BALANCED_ACTION_100

    async def start(self, observation, extra=None):
        t, _ = observation
        return self._get_action(t), {}

    async def step(self, reward: float, observation, extra):
        t, _ = observation
        return self._get_action(t), {}

    async def end(self, reward: float, extra):
        return {}
