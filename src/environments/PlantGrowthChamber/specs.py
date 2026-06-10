from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import jax.numpy as jnp
import joblib
import numpy as np
import pandas as pd

from environments.PlantGrowthChamber.utils import get_one_hot_time_observation
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, DIM_ACTION, RED_ACTION
from utils.functions import normalize
from utils.metrics import UnbiasedExponentialMovingAverage, iqm

WALL_STATS_COLS = [
    "wall_time",
    "clean_area",
    "clean_convex_hull_area",
    "clean_solidity",
    "clean_perimeter",
    "clean_width",
    "clean_height",
    "clean_longest_path",
    "clean_center_of_mass_x",
    "clean_center_of_mass_y",
    "clean_convex_hull_vertices",
    "clean_ellipse_center_x",
    "clean_ellipse_center_y",
    "clean_ellipse_major_axis",
    "clean_ellipse_minor_axis",
    "clean_ellipse_angle",
    "clean_ellipse_eccentricity",
    "clean_blue-yellow_frequencies_mean",
    "clean_blue_frequencies_mean",
    "clean_green-magenta_frequencies_mean",
    "clean_green_frequencies_mean",
    "clean_hue_circular_mean",
    "clean_hue_circular_std",
    "clean_hue_frequencies_mean",
    "clean_lightness_frequencies_mean",
    "clean_red_frequencies_mean",
    "clean_saturation_frequencies_mean",
    "clean_value_frequencies_mean",
    "log_clean_area",
    "days_since_sterilization",
    "days_since_transplant",
    "days_since_dome_removal",
    "days_since_watering",
    "liters_per_pot",
    "red_coef_trace_0.9",
    "white_coef_trace_0.9",
    "blue_coef_trace_0.9",
    "reward",
]


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


def _mean_clean_area(df: pd.DataFrame) -> float:
    if df.empty or "clean_area" not in df.columns:
        return 0.0
    return float(df["clean_area"].mean())


def _hours_normalized(local_time: datetime) -> float:
    hours_since_start = (local_time.hour - 9) + ((local_time.minute - 30) / 60)
    return float(np.clip(hours_since_start / 11.0, 0, 1))


@dataclass(frozen=True)
class AreaObservation(ObservationSpec):
    name: str = "area"
    shape: tuple[int, ...] = (1,)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        _, _, df = raw
        return np.array([_mean_clean_area(df)])


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
        mean_clean_area = _mean_clean_area(df)
        local_time = epoch_time.astimezone(backend.tz)
        hours_normalized = _hours_normalized(local_time)
        normalized_mean_clean_area = np.clip(normalize(mean_clean_area, 0, 680), 0, 0.9999)
        return np.array([hours_normalized, normalized_mean_clean_area])


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
        hours_normalized = _hours_normalized(local_time)
        return np.array([hours_normalized, dli])


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
            mean_clean_area = float(iqm(jnp.asarray(df["clean_area"]), 0.25, 0.9))
        else:
            mean_clean_area = 0.0

        if self.normalize_values:
            normalized_area = (mean_clean_area - self.clean_area_min) / (
                self.clean_area_max - self.clean_area_min
            )
        else:
            normalized_area = mean_clean_area

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


class WallStatsEmbeddingObservation(ObservationSpec):
    def __init__(self, trace_dim: int = 6):
        self.name = "wall_stats_embedding"
        self.trace_dim = trace_dim
        self.shape = (38 + 10 + 768,)
        self.action_uema: UnbiasedExponentialMovingAverage | None = None
        self.start_date: datetime | None = None
        self.embedding_dim = 768
        self.pca_dim = 10
        self.pca = None
        self.last_log_clean_area = None
        self.sterilization_date = None
        self.transplant_date = None
        self.dome_removal_date = None
        self.watering_date = None
        self.liters_per_pot = 0.0

    def setup(self, backend: Any, env_params: dict[str, Any]) -> None:
        beta = env_params.get("action_uema_beta", 0.9)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(self.trace_dim,), alpha=1 - beta
        )
        self.start_date = datetime.fromisoformat(env_params["start_date"])
        self.embedding_dim = env_params.get("embedding_dim", 768)
        self.pca_dim = env_params.get("pca_dim", 10)
        self.sterilization_date = env_params["sterilization_date"]
        self.transplant_date = env_params["transplant_date"]
        self.dome_removal_date = env_params["dome_removal_date"]
        self.watering_date = env_params["watering_date"]
        self.liters_per_pot = env_params["liters_per_pot"]
        pca_path = env_params.get(
            "pca_model_path",
            "/home/steven/Github/plant-data/results/v23/pca_model.joblib",
        )
        self.pca = joblib.load(pca_path)

    async def encode(
        self, raw: tuple[datetime, Any, pd.DataFrame], backend: Any
    ) -> np.ndarray:
        _, _, df = raw
        local_time = backend.get_local_time()
        assert self.start_date is not None
        assert self.action_uema is not None
        assert self.pca is not None

        wall_time = (
            (local_time - self.start_date.astimezone(backend.tz)).total_seconds()
            / 60
            / 60
            / 24
        )

        def days_since(date):
            if date is None:
                return 0.0
            if isinstance(date, str):
                date = datetime.fromisoformat(date).replace(tzinfo=backend.tz)
            return (local_time - date).total_seconds() / 60 / 60 / 24

        action_trace = self.action_uema.compute().flatten()

        if df.empty:
            mean_stats = np.zeros(len(WALL_STATS_COLS), dtype=np.float32)
            mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            cls_token_pca = np.zeros(self.pca_dim, dtype=np.float32)
        else:
            df = df.copy()
            df["wall_time"] = wall_time
            df["log_clean_area"] = np.log(df["clean_area"] + 1)
            df["days_since_sterilization"] = days_since(self.sterilization_date)
            df["days_since_transplant"] = days_since(self.transplant_date)
            df["days_since_dome_removal"] = days_since(self.dome_removal_date)
            df["days_since_watering"] = days_since(self.watering_date)
            df["liters_per_pot"] = self.liters_per_pot
            df["red_coef_trace_0.9"] = action_trace[0]
            df["white_coef_trace_0.9"] = action_trace[1]
            df["blue_coef_trace_0.9"] = action_trace[2]

            mean_log_clean_area = df["log_clean_area"].mean()
            if self.last_log_clean_area is not None:
                reward = mean_log_clean_area - self.last_log_clean_area
            else:
                reward = 0.0
            self.last_log_clean_area = mean_log_clean_area
            df["reward"] = reward

            q1 = df["clean_area"].quantile(0.25)
            q3 = df["clean_area"].quantile(0.75)
            mask = (
                (df["clean_area"] > q1)
                & (df["clean_area"] < q3)
                & ~np.isnan(df["clean_area"])
            )
            mean_stats = df[WALL_STATS_COLS].to_numpy(dtype=np.float32)
            mean_stats = np.nanmean(mean_stats[mask], axis=0)

            mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            cls_token_pca = np.zeros(self.pca_dim, dtype=np.float32)

            if "cls_token" in df.columns:
                mask_and_has_embedding = mask & ~df["cls_token"].isna()
                if mask_and_has_embedding.any():
                    stacked = np.stack(df["cls_token"][mask_and_has_embedding])
                    mean_embedding = np.mean(stacked, axis=0)
                    cls_token_pca = self.pca.transform(
                        mean_embedding.reshape(1, -1)
                    ).flatten()

        return np.concatenate((mean_stats, cls_token_pca, mean_embedding)).astype(
            np.float32
        )

    def update_action_trace(self, action: Any, backend: Any) -> None:
        assert self.action_uema is not None
        self.action_uema.update(jnp.array(action))


@dataclass(frozen=True)
class PPFD6Action(ActionSpec):
    name: str = "ppfd6"
    n_actions: int = 6
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        return np.asarray(action, dtype=np.float64)


@dataclass(frozen=True)
class IntensityAction(ActionSpec):
    name: str = "intensity"
    n_actions: int = 1
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        if isinstance(action, np.ndarray) and action.ndim > 0:
            return np.asarray(action, dtype=np.float64)
        return BALANCED_ACTION_105 * float(action)


@dataclass(frozen=True)
class DiscreteAction(ActionSpec):
    name: str = "discrete"
    n_actions: int = 2
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        if isinstance(action, np.ndarray):
            return np.asarray(action, dtype=np.float64)
        action_map = {0: DIM_ACTION, 1: BALANCED_ACTION_105}
        return action_map[int(action)]


@dataclass(frozen=True)
class ColorAction(ActionSpec):
    name: str = "color"
    n_actions: int = 3
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        if isinstance(action, np.ndarray):
            return np.asarray(action, dtype=np.float64)
        action_map = {
            0: BALANCED_ACTION_105,
            1: BLUE_ACTION,
            2: RED_ACTION,
        }
        return action_map[int(action)]


@dataclass(frozen=True)
class ColorTriangleAction(ActionSpec):
    name: str = "color_triangle"
    n_actions: int = 3
    trace_dim: int = 3

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] == 6:
            return action
        basis = np.column_stack([RED_ACTION, BALANCED_ACTION_105, BLUE_ACTION])
        return basis @ action


ACTION_SPECS: dict[str, ActionSpec] = {
    "ppfd6": PPFD6Action(),
    "intensity": IntensityAction(),
    "discrete": DiscreteAction(),
    "color": ColorAction(),
    "color_triangle": ColorTriangleAction(),
}


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
