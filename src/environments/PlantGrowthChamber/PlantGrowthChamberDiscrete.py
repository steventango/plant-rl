import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class PlantGrowthChamberDiscrete(PlantGrowthChamber):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
        self.action_map = {
            0: reference_spectrum * 0.675,
            1: reference_spectrum,
        }

    async def step(self, action: int):
        action = self.action_map[action]
        return await super().step(action)
