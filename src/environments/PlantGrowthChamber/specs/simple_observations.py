from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from environments.PlantGrowthChamber.specs.base import ObservationSpec
from environments.PlantGrowthChamber.utils import (
    get_one_hot_time_observation,
    hours_normalized,
    mean_clean_area,
)
from utils.functions import normalize


@dataclass(frozen=True)
class AreaObservation(ObservationSpec):
    name: str = "area"
    shape: tuple[int, ...] = (1,)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        _, _, df = raw
        return np.array([mean_clean_area(df)])


@dataclass(frozen=True)
class TimestampObservation(ObservationSpec):
    name: str = "timestamp"
    shape: tuple[int, ...] = (1,)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        epoch_time, _, _ = raw
        return np.array([epoch_time.timestamp()])


@dataclass(frozen=True)
class OneHotTimeObservation(ObservationSpec):
    name: str = "one_hot_time"
    shape: tuple[int, ...] = (13,)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        epoch_time, _, _ = raw
        local_time = epoch_time.astimezone(backend.tz)
        return get_one_hot_time_observation(local_time)


@dataclass(frozen=True)
class TimeAreaObservation(ObservationSpec):
    name: str = "time_area"
    shape: tuple[int, ...] = (2,)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        epoch_time, _, df = raw
        area = mean_clean_area(df)
        local_time = epoch_time.astimezone(backend.tz)
        normalized_hours = hours_normalized(local_time)
        normalized_area = np.clip(normalize(area, 0, 680), 0, 0.9999)
        return np.array([normalized_hours, normalized_area])


@dataclass(frozen=True)
class TimeDLIObservation(ObservationSpec):
    name: str = "time_dli"
    shape: tuple[int, ...] = (2,)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        epoch_time, _, _ = raw
        dli = np.clip(normalize(backend.dli, 0, 660), 0, 1)
        local_time = epoch_time.astimezone(backend.tz)
        normalized_hours = hours_normalized(local_time)
        return np.array([normalized_hours, dli])
