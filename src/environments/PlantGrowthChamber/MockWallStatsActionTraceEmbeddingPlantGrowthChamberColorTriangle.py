from environments.PlantGrowthChamber.MockPlantGrowthChamber import (
    MockPlantGrowthChamber,
)
from environments.PlantGrowthChamber.WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle import (
    WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle,
)


class MockWallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle(  # type: ignore
    MockPlantGrowthChamber, WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
