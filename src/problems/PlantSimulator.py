from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import PlantSimulator as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging 

class PlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params, seed = self.seed)  # default is using only the first plant and two actions (off and on)
        self.actions = 2
        self.observations = (5,)
        self.gamma = 0.99