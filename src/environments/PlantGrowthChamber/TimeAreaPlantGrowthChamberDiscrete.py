from environments.PlantGrowthChamber.TimeAreaPlantGrowthChamber import (
    TimeAreaPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import (
    PlantGrowthChamberDiscrete,
)


class TimeAreaPlantGrowthChamberDiscrete(  # type: ignore
    TimeAreaPlantGrowthChamber, PlantGrowthChamberDiscrete
):
    def __init__(self, *args, **kwargs):
        TimeAreaPlantGrowthChamber.__init__(self, *args, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, *args, **kwargs)
