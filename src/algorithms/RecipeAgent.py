import numpy as np
from datetime import datetime
from utils.RlGlue.agent import BaseAsyncAgent
from utils.constants import BALANCED_ACTION_100
from utils.checkpoint import checkpointable

@checkpointable(("start_date",))
class RecipeAgent(BaseAsyncAgent):
    def __init__(self, observations, actions, params, collector, seed):
        self._init_args = (observations, actions, params, collector, seed)
        self.params = params
        agent_params = params.get("agent", {})
        self.action_days = agent_params.get("action_days", [0])
        self.action_inputs = agent_params.get("action_inputs", [0.0])
        self.start_hour = 9
        self.end_hour = 21
        self.start_date = None

    def _is_night(self, t: datetime) -> bool:
        return t.hour >= self.end_hour or t.hour < self.start_hour

    def _is_morning(self, t: datetime) -> bool:
        return t.hour == self.start_hour and t.minute == 0

    def _get_ppfd(self, t: datetime) -> float:
        if self.start_date is None:
            return float(self.action_inputs[0])
        day_number = (t.date() - self.start_date).days
        ppfd = float(self.action_inputs[0])
        for day, value in zip(self.action_days, self.action_inputs):
            if day_number >= day:
                ppfd = float(value)
            else:
                break
        return ppfd

    def _get_action(self, t: datetime) -> float | np.ndarray:
        if self._is_night(t):
            return np.zeros(6)
        if self._is_morning(t):
            return 0.5 * BALANCED_ACTION_100
        return self._get_ppfd(t)

    async def start(self, observation: datetime, extra=None):
        self.start_date = observation.date()
        return self._get_action(observation), {}

    async def step(self, reward: float, observation: datetime, extra):
        return self._get_action(observation), {}

    async def end(self, reward: float, extra):
        return {}
