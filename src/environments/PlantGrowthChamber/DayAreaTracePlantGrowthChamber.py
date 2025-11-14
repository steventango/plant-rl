import jax.numpy as jnp
import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import UnbiasedExponentialMovingAverage, iqm

class DayAreaTracePlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = kwargs.get("normalize", True)
        self.action_uema_alpha = kwargs.get("action_uema_alpha", 0.9)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(6,), alpha=self.action_uema_alpha
        )
        self.start_date = self.get_local_time().date()
        self.day_min = 0.0
        self.day_max = 14.0
        self.clean_area_min = 14.3125
        self.clean_area_max = 1211.0

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await PlantGrowthChamber.get_observation(self)
        if not df.empty:
            #mean_clean_area = df["clean_area"].mean() if "clean_area" in df else 0.0
            mean_clean_area = iqm(jnp.asarray(df['clean_area']), 0.3, 0.1)
        else:
            mean_clean_area = 0.0
        if self.normalize:
            normalized_area = (mean_clean_area - self.clean_area_min) / (
                self.clean_area_max - self.clean_area_min
            )
        else:
            normalized_area = mean_clean_area
        day = (self.get_local_time().date() - self.start_date).days
        if self.normalize:
            normalized_day = (day - self.day_min) / (self.day_max - self.day_min)
        else:
            normalized_day = day

        action_trace = self.action_uema.compute()
        return np.array([normalized_day, normalized_area, *action_trace])

    def update_action_trace(self, action):
        self.action_uema.update(jnp.array(action)[None])
