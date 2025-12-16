from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle import (
    WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BasePlantGrowthChamberAsyncProblem import (
    BasePlantGrowthChamberAsyncProblem,
)


class WallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle(BasePlantGrowthChamberAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 3
        self.observations = (5,)
