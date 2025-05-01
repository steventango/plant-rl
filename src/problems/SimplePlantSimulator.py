from PyExpUtils.collection.Collector import Collector
from environments.SimplePlantSimulator import SimplePlantSimulator as DefaultEnv
from environments.SimplePlantSimulator import BanditEnv
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class SimplePlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        if 'type' not in self.env_params: 
            self.env = DefaultEnv(**self.env_params)
            self.actions = 4
            self.observations = (3,)
            self.gamma = 1.0
        elif self.env_params['type'] == 'BanditEnv':
            self.env = BanditEnv(**self.env_params)
            self.actions = 4
            self.observations = (1,)
            self.gamma = 0.0
        else:
            raise ValueError(f"{self.env_params['type']} is an invalid argument for SimplePlantSimulator type.")