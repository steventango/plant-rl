from environments.PlantGrowthChamber.AreaPlantGrowthChamber import (  # type: ignore
    AreaPlantGrowthChamber,
)
from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import (
    PlantGrowthChamberIntensity,
)


class MockAreaPlantGrowthChamberIntensity(  # type: ignore
    MockPlantGrowthChamber, AreaPlantGrowthChamber, PlantGrowthChamberIntensity
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
