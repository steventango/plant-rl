import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class PlantGrowthChamberDiscrete(PlantGrowthChamber):

    def __init__(self, zone: int, **kwargs):
        super().__init__(zone, **kwargs)
        reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
        self.action_map = {
            0: reference_spectrum * 0.675,
            1: reference_spectrum,
        }

    def step_one(self, action: int):
        action = self.action_map[action]
        return super().step_one(action)
