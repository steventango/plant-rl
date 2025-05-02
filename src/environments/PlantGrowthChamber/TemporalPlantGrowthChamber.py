import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class TemporalPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, zone: int, **kwargs):
        super().__init__(zone, **kwargs)

    def get_observation(self):
        super().get_observation()
        return np.array([self.time])
