from PyExpUtils.collection.Collector import Collector
from environments.SimplePlantSimulator import SimplePlantSimulator as SimEnv
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging

class SimplePlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = SimEnv(**self.env_params)
        self.actions = 4
        self.observations = (2,)
        self.gamma = 0.99