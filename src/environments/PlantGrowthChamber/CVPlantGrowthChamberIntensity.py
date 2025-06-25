import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import PlantGrowthChamberIntensity

class CVPlantGrowthChamberIntensity(PlantGrowthChamberIntensity):
    def __init__(self, **kwargs):
        PlantGrowthChamberIntensity.__init__(self, **kwargs)

class CVPlantGrowthChamberIntensity_MotionTracking(CVPlantGrowthChamberIntensity):
    async def get_observation(self):
        epoch_time, _, df = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)

        if not df.empty:
            mean_clean_area = df["clean_area"].mean() if "clean_area" in df else 0.0
        else:
            mean_clean_area = 0.0

        return np.array([local_time, mean_clean_area])