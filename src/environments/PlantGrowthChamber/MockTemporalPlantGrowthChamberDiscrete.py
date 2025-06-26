from environments.PlantGrowthChamber.MockPlantGrowthChamber import (  # type: ignore
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import (
    PlantGrowthChamberDiscrete,
)
from environments.PlantGrowthChamber.TemporalPlantGrowthChamber import (
    TemporalPlantGrowthChamber,
)


class MockTemporalPlantGrowthChamberDiscrete(  # type: ignore
    MockPlantGrowthChamber, PlantGrowthChamberDiscrete, TemporalPlantGrowthChamber
):
    def __init__(self, *args, **kwargs):
        MockPlantGrowthChamber.__init__(self, *args, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, *args, **kwargs)
