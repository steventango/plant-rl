import numpy as np  # type: ignore

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class TemporalPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_observation(self):  # type: ignore
        await super().get_observation()
        return np.array([self.time.timestamp()])
