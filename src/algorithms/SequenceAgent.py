import json
from datetime import date, datetime
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class SequenceAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.start_date: date = date.fromisoformat(params["local_start_date"])
        self.tz = ZoneInfo(params.get("timezone", "Etc/UTC"))

        if "action_days" in params:
            self._action_days: List[int] = params["action_days"]
            self._action_inputs: List = params["action_inputs"]
            self._dense: List[np.ndarray] = []
        else:
            self._action_days = []
            self._action_inputs = []
            self._dense = [
                np.array(a) for a in json.loads(params["actions"])
            ]

    def _action_for_day(self, day_number: int) -> np.ndarray:
        if self._action_days:
            action = self._action_inputs[0]
            for day, value in zip(self._action_days, self._action_inputs):
                if day_number >= day:
                    action = value
                else:
                    break
            return np.array(action)
        idx = min(day_number - 1, len(self._dense) - 1)
        return self._dense[idx]

    def _day_number(self, extra: Dict[str, Any]) -> int:
        local_date = datetime.fromtimestamp(extra["env_time"], tz=self.tz).date()
        return (local_date - self.start_date).days + 1

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._action_for_day(self._day_number(extra)), {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        return self._action_for_day(self._day_number(extra)), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
