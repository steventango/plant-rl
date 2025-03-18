import numpy as np

from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import PlantGrowthChamberIntensity


class CVPlantGrowthChamberIntensity(CVPlantGrowthChamber, PlantGrowthChamberIntensity):

    def __init__(self, **kwargs):
        CVPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberIntensity.__init__(self, zone=kwargs['zone'])
