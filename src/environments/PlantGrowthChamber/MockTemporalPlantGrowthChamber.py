from environments.PlantGrowthChamber.MockPlantGrowthChamber import (  # type: ignore
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.TemporalPlantGrowthChamber import (
    TemporalPlantGrowthChamber,
)


class MockTemporalPlantGrowthChamber(  # type: ignore
    MockPlantGrowthChamber, TemporalPlantGrowthChamber
):
    def __init__(self, *args, **kwargs):
        MockPlantGrowthChamber.__init__(self, *args, **kwargs)
