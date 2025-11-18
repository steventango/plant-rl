from environments.PlantGrowthChamber.DayAreaTracePlantGrowthChamber import (
    DayAreaTracePlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberColorTriangle import (
    PlantGrowthChamberColorTriangle,
)
from utils.metrics import UnbiasedExponentialMovingAverage


class DayAreaTracePlantGrowthChamberColorTriangle(  # type: ignore
    DayAreaTracePlantGrowthChamber, PlantGrowthChamberColorTriangle
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(3,), alpha=1 - self.action_uema_beta
        )
