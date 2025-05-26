import numpy as np

from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import OfflinePlantGrowthChamber
from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import PlantGrowthChamberDiscrete


class OfflineCVPlantGrowthChamberDiscrete(OfflinePlantGrowthChamber, CVPlantGrowthChamber, PlantGrowthChamberDiscrete):

    def __init__(self, *args, **kwargs):
        OfflinePlantGrowthChamber.__init__(self, *args, **kwargs)
        CVPlantGrowthChamber.__init__(self, *args, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, *args, **kwargs)
