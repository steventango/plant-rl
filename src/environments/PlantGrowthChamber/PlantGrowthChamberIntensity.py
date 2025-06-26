import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class PlantGrowthChamberIntensity(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])

    async def step(self, action: float | np.ndarray):
        if isinstance(action, np.ndarray):
            return await super().step(action)
        action = self.reference_spectrum * action
        return await super().step(action)
