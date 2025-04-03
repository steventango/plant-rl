import numpy as np

from environments.PlantGrowthChamber.MockPlantGrowthChamber import MockPlantGrowthChamber
from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import PlantGrowthChamberDiscrete


class MockCVPlantGrowthChamberDiscrete(MockPlantGrowthChamber, CVPlantGrowthChamber, PlantGrowthChamberDiscrete):

    def __init__(self, **kwargs):
        MockPlantGrowthChamber.__init__(self, zone=kwargs['zone'], path=kwargs['path'])
        CVPlantGrowthChamber.__init__(self, zone=kwargs['zone'], total_steps=kwargs.get('total_steps'))
        PlantGrowthChamberDiscrete.__init__(self, zone=kwargs['zone'])
