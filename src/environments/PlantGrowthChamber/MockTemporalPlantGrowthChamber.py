from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.TemporalPlantGrowthChamber import (
    TemporalPlantGrowthChamber,
)


class MockTemporalPlantGrowthChamber(MockPlantGrowthChamber, TemporalPlantGrowthChamber):

    def __init__(self, **kwargs):
        MockPlantGrowthChamber.__init__(self, zone=kwargs["zone"], path=kwargs["path"])
        TemporalPlantGrowthChamber.__init__(self, zone=kwargs["zone"])
