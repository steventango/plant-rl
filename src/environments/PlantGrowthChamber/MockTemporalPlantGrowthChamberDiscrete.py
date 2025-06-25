from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import (
    PlantGrowthChamberDiscrete,
)
from environments.PlantGrowthChamber.TemporalPlantGrowthChamber import (
    TemporalPlantGrowthChamber,
)


class MockTemporalPlantGrowthChamberDiscrete(
    MockPlantGrowthChamber, PlantGrowthChamberDiscrete, TemporalPlantGrowthChamber
):
    def __init__(self, *args, **kwargs):
        MockPlantGrowthChamber.__init__(self, *args, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, *args, **kwargs)
