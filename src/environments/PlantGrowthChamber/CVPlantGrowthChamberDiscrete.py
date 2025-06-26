from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber  # type: ignore
from environments.PlantGrowthChamber.PlantGrowthChamberDiscrete import (
    PlantGrowthChamberDiscrete,
)


class CVPlantGrowthChamberDiscrete(CVPlantGrowthChamber, PlantGrowthChamberDiscrete):  # type: ignore
    def __init__(self, **kwargs):
        CVPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberDiscrete.__init__(self, **kwargs)
