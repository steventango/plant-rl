import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.functions import normalize


class TimeDLIPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        dli = np.clip(normalize(self.dli, 0, 66), 0, 1)
        local_time = epoch_time.astimezone(self.tz)
        hours_since_start = (local_time.hour - 9) + ((local_time.minute - 30) / 60)
        hours_normalized = np.clip(hours_since_start / 11.0, 0, 1)
        return np.array([hours_normalized, dli])
