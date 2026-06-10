from __future__ import annotations

from typing import Any

from environments.PlantGrowthChamber.specs.base import ActionSpec, ObservationSpec
from environments.PlantGrowthChamber.specs.day_area_trace import DayAreaTraceObservation
from environments.PlantGrowthChamber.specs.simple_observations import (
    AreaObservation,
    OneHotTimeObservation,
    TimeAreaObservation,
    TimeDLIObservation,
    TimestampObservation,
)
from environments.PlantGrowthChamber.specs.wall_stats import WallStatsEmbeddingObservation


def create_observation_spec(
    name: str, action_spec: ActionSpec, env_params: dict[str, Any]
) -> ObservationSpec:
    if name in ("scalar", "area"):
        return AreaObservation()
    if name == "timestamp":
        return TimestampObservation()
    if name == "one_hot_time":
        return OneHotTimeObservation()
    if name == "time_area":
        return TimeAreaObservation()
    if name == "time_dli":
        return TimeDLIObservation()
    if name == "day_area_trace":
        return DayAreaTraceObservation(trace_dim=action_spec.trace_dim)
    if name == "wall_stats_embedding":
        return WallStatsEmbeddingObservation(trace_dim=action_spec.trace_dim)
    raise ValueError(f"Unknown observation spec: {name}")
