from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import (
    PlantGrowthChamberIntensity as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BaseAsyncProblem import BaseAsyncProblem


class exp1_iter1(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int):
        super().__init__(exp, idx)
        self.env = Env(**self.env_params)
        self.actions = 1
        self.observations = (1,)
