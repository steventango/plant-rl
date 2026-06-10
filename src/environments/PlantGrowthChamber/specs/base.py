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
