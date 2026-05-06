from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.TemporalPlantGrowthChamberColorTriangle import (
    TemporalPlantGrowthChamberColorTriangle,
)


class MockTemporalPlantGrowthChamberColorTriangle(
    MockPlantGrowthChamber, TemporalPlantGrowthChamberColorTriangle
):
    def __init__(self, *args, **kwargs):
        MockPlantGrowthChamber.__init__(self, *args, **kwargs)
