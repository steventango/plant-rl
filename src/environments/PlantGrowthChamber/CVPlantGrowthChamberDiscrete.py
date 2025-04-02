import numpy as np

from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import PlantGrowthChamberDiscrete


class CVPlantGrowthChamberDiscrete(CVPlantGrowthChamber, PlantGrowthChamberDiscrete):

    def __init__(self, **kwargs):
        CVPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, zone=kwargs['zone'])
