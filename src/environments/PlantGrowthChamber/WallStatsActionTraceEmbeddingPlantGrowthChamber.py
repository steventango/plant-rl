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
        self.start_date = self.get_local_time().date()
        self.embedding_dim = kwargs.get("embedding_dim", 768)

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await PlantGrowthChamber.get_observation(self)

        # 1. Wall Time
        wall_time = (self.get_local_time().date() - self.start_date).total_seconds() / 60 / 60 / 24

        # TODO: clean the stats
        # concat df columns into a single array
        df.drop('area', axis=1, inplace=True)
        df.columns = ["clean_" + col if not col.startswith("clean_") else col for col in df.columns]
        clean_stats = df[COLS].to_numpy(dtype=np.float32)
        # take the mean across plants
        #TODO Mask out dead plants
        mean_clean_stats = np.nanmean(clean_stats, axis=0)

        # 3. Mean Embedding
        mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        if not df.empty and "cls_token" in df.columns:
            # Filter out valid embeddings (lists/arrays)
            valid_embeddings = [
                e
                for e in df["cls_token"]
                if e is not None and isinstance(e, (list, np.ndarray)) and len(e) > 0
            ]
            if valid_embeddings:
                # Stack and compute mean
                stacked = np.stack(valid_embeddings)
                current_mean = np.mean(stacked, axis=0)

                # Check dimension
                if current_mean.shape[0] == self.embedding_dim:
                    mean_embedding = current_mean

        # 4. Action Trace (Area Trace)
        action_trace = self.action_uema.compute().flatten()

        
        # Concatenate
        observation = np.concatenate(([wall_time], mean_clean_stats, action_trace, mean_embedding))
        return observation

    def update_action_trace(self, action):
        self.action_uema.update(jnp.array(action))
