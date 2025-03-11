import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class PlantGrowthChamberIntensity(PlantGrowthChamber):

    def __init__(self, zone: int):
        super().__init__(zone)
        self.reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])

    def put_action(self, action: float):
        action = self.reference_spectrum * action
        return super().put_action(action)
