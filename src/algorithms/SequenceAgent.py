import json
from datetime import date, datetime
from typing import Any, Dict, Tuple
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
        self.actions = [
            np.array(action) for action in json.loads(self.params["actions"])
        ]
        self.start_date: date = date.fromisoformat(params["local_start_date"])
        self.tz = ZoneInfo(params.get("timezone", "Etc/UTC"))

    def _day_number(self, extra: Dict[str, Any]) -> int:
        local_date = datetime.fromtimestamp(extra["env_time"], tz=self.tz).date()
        return (local_date - self.start_date).days + 1

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        idx = min(self._day_number(extra) - 1, len(self.actions) - 1)
        return self.actions[idx], {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        idx = min(self._day_number(extra) - 1, len(self.actions) - 1)
        return self.actions[idx], {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}
