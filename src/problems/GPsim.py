from PyExpUtils.collection.Collector import Collector

from environments.GPsim.GPsim_1day import GPsim_1day
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class GPsim(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.actions = 3
        self.observations = (5,)
        self.gamma = 0.99
        self.env = GPsim_1day(**self.env_params)
