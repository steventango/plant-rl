from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber  # type: ignore
from environments.PlantGrowthChamber.PlantGrowthChamberColour import (
    PlantGrowthChamberColour,
)


class CVPlantGrowthChamberDiscrete(CVPlantGrowthChamber, PlantGrowthChamberColour):  # type: ignore
    def __init__(self, **kwargs):
        CVPlantGrowthChamber.__init__(self, **kwargs)
        PlantGrowthChamberColour.__init__(self, **kwargs)
