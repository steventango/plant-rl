from environments.PlantGrowthChamber.AreaPlantGrowthChamber import (
    AreaPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import (
    PlantGrowthChamberIntensity,
)


class AreaPlantGrowthChamberIntensity(
    AreaPlantGrowthChamber, PlantGrowthChamberIntensity
):
    def __init__(self, **kwargs):
        AreaPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberIntensity.__init__(self, **kwargs)
