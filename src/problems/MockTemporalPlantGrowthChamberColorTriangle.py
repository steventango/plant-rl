from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.MockTemporalPlantGrowthChamberColorTriangle import (
    MockTemporalPlantGrowthChamberColorTriangle as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BasePlantGrowthChamberAsyncProblem import (
    BasePlantGrowthChamberAsyncProblem,
)


class MockTemporalPlantGrowthChamberColorTriangle(BasePlantGrowthChamberAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 3
        self.observations = (1,)
