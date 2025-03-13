import numpy as np

from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import PlantGrowthChamberIntensity


class CVPlantGrowthChamberIntensity(CVPlantGrowthChamber, PlantGrowthChamberIntensity):

    def __init__(self, zone: int):
        super().__init__(zone)

