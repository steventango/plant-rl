from datetime import datetime
import joblib
import jax.numpy as jnp
import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import UnbiasedExponentialMovingAverage


COLS = [
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


class WallStatsActionTraceEmbeddingPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_uema_beta = kwargs.get("action_uema_beta", 0.9)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(6,), alpha=1 - self.action_uema_beta
        )
        self.start_date = datetime.fromisoformat(kwargs["start_date"])
        self.embedding_dim = kwargs.get("embedding_dim", 768)
        self.pca_dim = kwargs.get("pca_dim", 10)

        # Dates for days_since attributes
        self.sterilization_date = kwargs["sterilization_date"]
        self.transplant_date = kwargs["transplant_date"]
        self.dome_removal_date = kwargs["dome_removal_date"]
        self.watering_date = kwargs["watering_date"]
        self.liters_per_pot = kwargs["liters_per_pot"]

        # Load PCA model
        pca_path = kwargs.get(
            "pca_model_path",
            "/home/steven/Github/plant-data/results/v23/pca_model.joblib",
        )
        self.pca = joblib.load(pca_path)
        self.last_log_clean_area = None

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await PlantGrowthChamber.get_observation(self)
        local_time = self.get_local_time()

        # 1. Wall Time
        wall_time = (local_time - self.start_date.astimezone(self.tz)).total_seconds() / 60 / 60 / 24

        def days_since(date):
            if date is None:
                return 0.0
            if isinstance(date, str):
                date = datetime.fromisoformat(date).replace(tzinfo=self.tz)
            return (local_time - date).total_seconds() / 60 / 60 / 24

        # Action traces
        action_trace = self.action_uema.compute().flatten()

        # if df is empty return all 0s
        if df.empty:
            mean_stats = np.zeros(len(COLS), dtype=np.float32)
            mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            cls_token_pca = np.zeros(self.pca_dim, dtype=np.float32)
        else:
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

            # Calculate reward (delta log clean area)
            mean_log_clean_area = df["log_clean_area"].mean()
            if self.last_log_clean_area is not None:
                reward = mean_log_clean_area - self.last_log_clean_area
            else:
                reward = 0.0
            self.last_log_clean_area = mean_log_clean_area
            df["reward"] = reward

            # Filter and take mean
            alive_mask = (df["clean_area"] > 0) & ~np.isnan(df["clean_area"])
            mean_stats = df[COLS].to_numpy(dtype=np.float32)
            mean_stats = np.nanmean(mean_stats[alive_mask], axis=0)

            # 3. Mean Embedding
            mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            cls_token_pca = np.zeros(self.pca_dim, dtype=np.float32)

            if "cls_token" in df.columns:
                alive_mask_and_has_embedding = alive_mask & ~df["cls_token"].isna()
                if alive_mask_and_has_embedding.any():
                    stacked = np.stack(df["cls_token"][alive_mask_and_has_embedding])
                    mean_embedding = np.mean(stacked, axis=0)
                    cls_token_pca = self.pca.transform(
                        mean_embedding.reshape(1, -1)
                    ).flatten()

        observation = np.concatenate(
            (mean_stats, cls_token_pca, mean_embedding)
        ).astype(np.float32)
        return observation

    def update_action_trace(self, action):
        self.action_uema.update(jnp.array(action))
