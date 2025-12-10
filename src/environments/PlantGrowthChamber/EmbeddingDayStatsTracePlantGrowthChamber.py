import jax.numpy as jnp
import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import UnbiasedExponentialMovingAverage


class EmbeddingDayStatsTracePlantGrowthChamber(PlantGrowthChamber):
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

        # 1. Day
        day = (self.get_local_time().date() - self.start_date).days

        # 2. Mean Clean Plant Stats
        mean_clean_area = 0.0
        if not df.empty and "clean_area" in df.columns:
            # We assume df is ordered by plant index corresponding to self.zone.num_plants
            # But process_plants returns a list based on pot_quads.
            clean_areas = df["clean_area"].to_numpy(dtype=np.float32)
            if clean_areas.size > 0:
                mean_clean_area = np.mean(clean_areas)

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
        action_trace = self.action_uema.compute()

        # Concatenate: Mean Embedding, Day, Mean Clean Plant Stats, Area Trace
        return np.concatenate(([day], [mean_clean_area], action_trace, mean_embedding))

    def update_action_trace(self, action):
        self.action_uema.update(jnp.array(action)[None])
