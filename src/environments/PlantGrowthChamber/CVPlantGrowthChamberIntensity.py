import numpy as np  # type: ignore

from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import (
    PlantGrowthChamberIntensity,
)


class CVPlantGrowthChamberIntensity(PlantGrowthChamberIntensity):
    def __init__(self, **kwargs):
        PlantGrowthChamberIntensity.__init__(self, **kwargs)


class CVPlantGrowthChamberIntensity_MotionTracking(CVPlantGrowthChamberIntensity):
    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)

        if not df.empty:
            total_area = df["area"].sum()
        else:
            total_area = 0.0

        return np.array([local_time, total_area])
