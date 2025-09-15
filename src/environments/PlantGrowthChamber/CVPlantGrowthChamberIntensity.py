import numpy as np  # type: ignore
from datetime import timedelta
from utils.constants import BALANCED_ACTION_100

from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import (
    PlantGrowthChamberIntensity,
)


class CVPlantGrowthChamberIntensity(PlantGrowthChamberIntensity):
    def __init__(self, **kwargs):
        PlantGrowthChamberIntensity.__init__(self, **kwargs)


class CVPlantGrowthChamberIntensity_MotionTracking(CVPlantGrowthChamberIntensity):
    def __init__(self, **kwargs):
        CVPlantGrowthChamberIntensity.__init__(self, **kwargs)
        self.duration = timedelta(minutes=5)
        self.reference_spectrum = BALANCED_ACTION_100

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)

        if not df.empty:
            clean_areas = df["clean_area"]
            mean_area = clean_areas.mean()
        else:
            mean_area = 0.0

        return [local_time, mean_area]
