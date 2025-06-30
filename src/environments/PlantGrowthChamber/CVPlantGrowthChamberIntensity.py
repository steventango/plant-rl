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
            clean_areas = df["clean_area"]
            # mean_area = clean_areas.mean()
            mean_area = np.mean(clean_areas[18:])
        else:
            mean_area = 0.0

        return np.array([local_time, mean_area])
