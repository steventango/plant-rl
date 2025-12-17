import jax.numpy as jnp
import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import UnbiasedExponentialMovingAverage


COLS = [
    # "wall_time",
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
    # "red_coef_trace_0.9",
    # "white_coef_trace_0.9",
    # "blue_coef_trace_0.9",
]


class WallStatsActionTraceEmbeddingPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_uema_beta = kwargs.get("action_uema_beta", 0.9)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(6,), alpha=1 - self.action_uema_beta
        )
        self.start_date = self.get_local_time().replace(hour=9, minute=30)
        self.embedding_dim = kwargs.get("embedding_dim", 768)

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await PlantGrowthChamber.get_observation(self)

        # 1. Wall Time
        wall_time = (
            (self.get_local_time() - self.start_date).total_seconds()
            / 60
            / 60
            / 24
        )

        # if df is empty return all 0s
        if df.empty:
            mean_clean_stats = np.zeros(len(COLS), dtype=np.float32)
            mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            clean_stats = df[COLS].to_numpy(dtype=np.float32)

            # take the mean across alive plants
            alive_mask = (df["clean_area"] > 0) & ~np.isnan(df["clean_area"])
            mean_clean_stats = np.nanmean(clean_stats[alive_mask], axis=0)

            # 3. Mean Embedding
            mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            alive_mask_and_has_embedding = alive_mask & ~df["cls_token"].isna()
            if not df.empty and "cls_token" in df.columns:
                stacked = np.stack(df["cls_token"][alive_mask_and_has_embedding])
                mean_embedding = np.mean(stacked, axis=0)

        # 4. Action Trace (Area Trace)
        action_trace = self.action_uema.compute().flatten()

        # Concatenate
        observation = np.concatenate(
            ([wall_time], mean_clean_stats, action_trace, mean_embedding)
        )
        return observation

    def update_action_trace(self, action):
        self.action_uema.update(jnp.array(action))
