from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.MockCVPlantGrowthChamberDiscrete import (
    MockCVPlantGrowthChamberDiscrete as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BaseAsyncProblem import BaseAsyncProblem


class MockCVPlantGrowthChamberDiscrete(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 10
        self.observations = (2,)
        self.gamma = 0.99
