from PyExpUtils.collection.Collector import Collector
from environments.SimplePlantSimulator import SimplePlantSimulator as DefaultEnv
from environments.SimplePlantSimulator import TrivialRewEnv, SineTimeEnv, TrivialRewSineTimeEnv
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging

class SimplePlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.gamma = 0.99
        if self.env_params['type'] == 'Default':
            self.env = DefaultEnv(**self.env_params)
            self.actions = 4
            self.observations = (2,)
        elif self.env_params['type'] == 'TrivialRew':
            self.env = TrivialRewEnv(**self.env_params)
            self.actions = 4
            self.observations = (2,)
        elif self.env_params['type'] == 'SineTime':
            self.env = SineTimeEnv(**self.env_params)
            self.actions = 4
            self.observations = (3,)
        elif self.env_params['type'] == 'TrivialRewSineTime':
            self.env = TrivialRewSineTimeEnv(**self.env_params)
            self.actions = 4
            self.observations = (3,)
        else:
            raise ValueError(f"{self.env_params['type']} is an invalid argument for SimplePlantSimulator type.")
        