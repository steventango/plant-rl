import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class TemporalPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_observation(self):
        await super().get_observation()
        return np.array([self.time.timestamp()])
