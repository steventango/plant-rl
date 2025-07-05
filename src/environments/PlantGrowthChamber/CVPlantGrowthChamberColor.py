from environments.PlantGrowthChamber.CVPlantGrowthChamber import (
    CVPlantGrowthChamber,  # type: ignore
)
from environments.PlantGrowthChamber.PlantGrowthChamberColor import (
    PlantGrowthChamberColor,
)


class CVPlantGrowthChamberColor(CVPlantGrowthChamber, PlantGrowthChamberColor):  # type: ignore
    def __init__(self, **kwargs):
        CVPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberColor.__init__(self, **kwargs)
