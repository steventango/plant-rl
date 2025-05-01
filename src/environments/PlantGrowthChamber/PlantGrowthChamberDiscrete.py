import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class PlantGrowthChamberDiscrete(PlantGrowthChamber):

    def __init__(self, zone: int):
        super().__init__(zone)
        reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
        red = np.array([0.000, 0.324, 0.324, 0.000, 1.000, 0.606])
        blue = np.array([1.000, 0.324, 0.324, 0.000, 0.000, 0.000])
        self.action_map = {
            0: reference_spectrum * 0.350,
            1: reference_spectrum * 0.675,
            2: reference_spectrum,
            3: reference_spectrum * 1.652,
            4: red * 0.675,
            5: red,
            6: red * 1.652,
            7: blue * 0.675,
            8: blue,
            9: blue * 1.652,
        }

    def put_action(self, action: int):
        action = self.action_map[action]
        return super().put_action(action)
