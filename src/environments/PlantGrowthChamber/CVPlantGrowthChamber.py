import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import UnbiasedExponentialMovingAverage as uema


class CVPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_observation(self):
        epoch_time, _, plant_stats = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        time_since_morning = epoch_time - morning_time
        seconds_since_morning = time_since_morning.total_seconds()
        time_progress = seconds_since_morning / (12 * 3600)
        return np.array([time_progress])
