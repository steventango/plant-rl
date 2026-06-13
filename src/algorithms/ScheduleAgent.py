from datetime import date, datetime
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.checkpoint import checkpointable


@checkpointable(("steps",))
class ScheduleAgent(BaseAgent):
    """Day-based color-schedule agent.

    Returns a scalar action in [-1, 1] (for use with the ``continuous_color``
    action spec) based on a look-up table keyed by experiment day number.

    Night / dawn / dusk enforcement is delegated to
    ``PlantGrowthChamberAsyncAgentWrapper``.

    Params
    ------
    action_days : list[int]
        Ascending list of day numbers on which the action changes.
    action_inputs : list[float]
        Action value to use from each corresponding day onward.
        -1 → all-blue, 0 → balanced, +1 → all-red.
    local_start_date : str
        ISO-8601 date string (YYYY-MM-DD) for experiment day 1.
    timezone : str
        IANA timezone name used to convert ``env_time`` from extra dict.
    """

    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        self.action_days: list[int] = params.get("action_days", [1])
        self.action_inputs: list[float] = params.get("action_inputs", [0.0])
        self.start_date: date = date.fromisoformat(
            params.get("local_start_date", "2000-01-01")
        )
        self.tz = ZoneInfo(params.get("timezone", "Etc/UTC"))

    def _get_scalar_action(self, local_date: date) -> float:
        day_number = (local_date - self.start_date).days + 1
        scalar = float(self.action_inputs[0])
        for day, value in zip(self.action_days, self.action_inputs, strict=False):
            if day_number >= day:
                scalar = float(value)
            else:
                break
        return scalar

    def _local_date_from_extra(self, extra: Dict[str, Any]) -> date:
        env_time = datetime.fromtimestamp(extra["env_time"], tz=self.tz)
        return env_time.date()

    def policy(self, observation: np.ndarray, deterministic: bool = True) -> float:
        return 0.0

    def start(
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        local_date = self._local_date_from_extra(extra)
        return self._get_scalar_action(local_date), {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        self.steps += 1
        local_date = self._local_date_from_extra(extra)
        return self._get_scalar_action(local_date), {}

    def end(self, reward: float, extra: Dict[str, Any]) -> Dict[str, Any]:
        return {}
