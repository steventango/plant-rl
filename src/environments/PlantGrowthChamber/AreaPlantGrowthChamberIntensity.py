from environments.PlantGrowthChamber.AreaPlantGrowthChamber import (  # type: ignore
    AreaPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import (
    PlantGrowthChamberIntensity,
)


class AreaPlantGrowthChamberIntensity(  # type: ignore
    AreaPlantGrowthChamber, PlantGrowthChamberIntensity
):
    def __init__(self, **kwargs):
        AreaPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberIntensity.__init__(self, **kwargs)
