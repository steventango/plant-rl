from environments.PlantGrowthChamber.WallStatsActionTraceEmbeddingPlantGrowthChamber import (
    WallStatsActionTraceEmbeddingPlantGrowthChamber,
)
from environments.PlantGrowthChamber.PlantGrowthChamberColorTriangle import (
    PlantGrowthChamberColorTriangle,
)
from utils.metrics import UnbiasedExponentialMovingAverage


class WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle(  # type: ignore
    WallStatsActionTraceEmbeddingPlantGrowthChamber, PlantGrowthChamberColorTriangle
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_uema = UnbiasedExponentialMovingAverage(
            shape=(3,), alpha=1 - self.action_uema_beta
        )
