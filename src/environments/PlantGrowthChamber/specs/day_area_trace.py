from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from datetime import datetime

from environments.PlantGrowthChamber.specs.base import ObservationSpec
from utils.metrics import UnbiasedExponentialMovingAverage, iqm


class DayAreaTraceObservation(ObservationSpec):
    def __init__(self, trace_dim: int = 6, normalize_values: bool = True):
        self.name = "day_area_trace"
        self.trace_dim = trace_dim
        self.shape = (2 + trace_dim,)
        self.normalize_values = normalize_values
        self.action_uema: UnbiasedExponentialMovingAverage | None = None
        self.start_date = None
        self.day_min = 0.0
        self.day_max = 14.0
        self.clean_area_min = 14.3125
        self.clean_area_max = 1211.0

    def setup(self, backend: Any, env_params: dict[str, Any]) -> None:
        beta = env_params.get("action_uema_beta", 0.9)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(self.trace_dim,), alpha=1 - beta
        )
        self.normalize_values = env_params.get("normalize", self.normalize_values)
        self.start_date = backend.get_local_time().date()

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        _, _, df = raw
        if not df.empty:
            area = float(iqm(jnp.asarray(df["clean_area"]), 0.25, 0.9))
        else:
            area = 0.0

        if self.normalize_values:
            normalized_area = (area - self.clean_area_min) / (
                self.clean_area_max - self.clean_area_min
            )
        else:
            normalized_area = area

        day = (backend.get_local_time().date() - self.start_date).days
        if self.normalize_values:
            normalized_day = (day - self.day_min) / (self.day_max - self.day_min)
        else:
            normalized_day = day

        assert self.action_uema is not None
        action_trace = self.action_uema.compute()
        return np.array([normalized_day, normalized_area, *action_trace])

    def update_action_trace(self, action: Any, backend: Any) -> None:
        assert self.action_uema is not None
        self.action_uema.update(jnp.array(action)[None])
