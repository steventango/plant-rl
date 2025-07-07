from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import (
    PlantGrowthChamberDiscrete,
)
from environments.PlantGrowthChamber.TimeDLIPlantGrowthChamber import (
    TimeDLIPlantGrowthChamber,
)


class TimeDLIPlantGrowthChamberDiscrete(  # type: ignore
    TimeDLIPlantGrowthChamber, PlantGrowthChamberDiscrete
):
    def __init__(self, *args, **kwargs):
        TimeDLIPlantGrowthChamber.__init__(self, *args, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, *args, **kwargs)
