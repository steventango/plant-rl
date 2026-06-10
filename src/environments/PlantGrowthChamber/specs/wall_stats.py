from __future__ import annotations

from datetime import datetime
from typing import Any

import jax.numpy as jnp
import joblib
import numpy as np
import pandas as pd

from environments.PlantGrowthChamber.specs.base import ObservationSpec
from utils.metrics import UnbiasedExponentialMovingAverage

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
