from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


class ObservationSpec(ABC):
    name: str
    shape: tuple[int, ...]

    def setup(self, backend: Any, env_params: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        pass

    def update_action_trace(self, action: Any, backend: Any) -> None:
        pass


class ActionSpec(ABC):
    name: str
    n_actions: int
    trace_dim: int = 6

    @abstractmethod
    def decode(self, action: Any, backend: Any) -> np.ndarray:
        pass


def mean_clean_area(df: pd.DataFrame) -> float:
    if df.empty or "clean_area" not in df.columns:
        return 0.0
    return float(df["clean_area"].mean())


def hours_normalized(local_time: datetime) -> float:
    hours_since_start = (local_time.hour - 9) + ((local_time.minute - 30) / 60)
    return float(np.clip(hours_since_start / 11.0, 0, 1))
