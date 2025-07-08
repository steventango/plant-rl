import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.functions import normalize


class TimeAreaPlantGrowthChamber(PlantGrowthChamber):
    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        if not df.empty:
            mean_clean_area = df["clean_area"].mean() if "clean_area" in df else 0.0
        else:
            mean_clean_area = 0.0
        local_time = epoch_time.astimezone(self.tz)
        hours_since_start = (local_time.hour - 9) + ((local_time.minute - 30) / 60)
        hours_normalized = hours_since_start / 11.0
        hours_normalized = np.clip(hours_normalized, 0, 1)
        normalized_mean_clean_area = normalize(mean_clean_area, 0, 680)
        normalized_mean_clean_area = np.clip(normalized_mean_clean_area, 0, 0.9999)
        return np.array([hours_normalized, normalized_mean_clean_area])
