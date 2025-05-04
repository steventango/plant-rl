from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.MockTemporalPlantGrowthChamberDiscrete import (
    MockTemporalPlantGrowthChamberDiscrete as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BaseAsyncProblem import BaseAsyncProblem


class MockTemporalPlantGrowthChamberDiscrete(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 2
        self.observations = (1,)
        self.gamma = 1
