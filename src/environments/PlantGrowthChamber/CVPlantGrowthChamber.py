import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class CVPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, zone: int):
        super().__init__(zone)

    def get_observation(self):
        super().get_observation()
        # todo: process stuff here...
        return np.array([self.time, 1, 2, 3, 4, 5])
