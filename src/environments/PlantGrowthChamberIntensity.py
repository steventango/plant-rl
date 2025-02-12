import numpy as np
from environments.PlantGrowthChamber import PlantGrowthChamber


class PlantGrowthChamberIntensity(PlantGrowthChamber):
    def __init__(self, camera_url: str, lightbar_url: str):
        super().__init__(camera_url, lightbar_url)
        self.reference_spectrum = np.array([0.199, 0.381, 0.162, 0.000, 0.166, 0.303])

    def put_action(self, action: float):
        action = self.reference_spectrum * action
        print(action)
        return super().put_action(action)
