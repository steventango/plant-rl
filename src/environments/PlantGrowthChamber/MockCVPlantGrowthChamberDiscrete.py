from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber
from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import (
    PlantGrowthChamberDiscrete,
)


class MockCVPlantGrowthChamberDiscrete(
    MockPlantGrowthChamber, CVPlantGrowthChamber, PlantGrowthChamberDiscrete
):
    def __init__(self, *args, **kwargs):
        MockPlantGrowthChamber.__init__(self, *args, **kwargs)
        CVPlantGrowthChamber.__init__(self, *args, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, *args, **kwargs)
