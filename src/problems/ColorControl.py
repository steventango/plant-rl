from environments.PlantGrowthChamber.PlantGrowthChamberColor import (
    PlantGrowthChamberColor as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BaseAsyncProblem import BaseAsyncProblem


class ColorControl(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector=None):
        super().__init__(exp, idx, collector)
        self.env = Env(**self.env_params)
        self.actions = 1
        self.observations = (2,)
