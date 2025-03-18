import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class TemporalPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, zone: int):
        super().__init__(zone)

    def get_observation(self):
        super().get_observation()
        return np.array([self.time])
