import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class AreaPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_observation(self):
        epoch_time, _, df = await super().get_observation()
        mean_clean_area = df["clean_area"].mean() if "clean_area" in df else 0.0
        return np.array([mean_clean_area])
