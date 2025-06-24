import numpy as np

from environments.PlantGrowthChamber.MockPlantGrowthChamber import MockPlantGrowthChamber
from environments.PlantGrowthChamber.AreaPlantGrowthChamber import AreaPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import PlantGrowthChamberIntensity


class MockAreaPlantGrowthChamberIntensity(MockPlantGrowthChamber, AreaPlantGrowthChamber, PlantGrowthChamberIntensity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
