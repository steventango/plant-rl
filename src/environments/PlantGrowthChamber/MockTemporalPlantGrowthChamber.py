from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.TemporalPlantGrowthChamber import (
    TemporalPlantGrowthChamber,
)


class MockTemporalPlantGrowthChamber(
    MockPlantGrowthChamber, TemporalPlantGrowthChamber
):
    def __init__(self, *args, **kwargs):
        MockPlantGrowthChamber.__init__(self, *args, **kwargs)
