from PyExpUtils.collection.Collector import Collector
from environments.SimplePlantSimulator import SimplePlantSimulator as DefaultEnv
from environments.SimplePlantSimulator import Daily_ContextBandit
from environments.SimplePlantSimulator import Daily_Bandit
from environments.SimplePlantSimulator import Daily_ESARSA_TOD
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class SimplePlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        if 'type' not in self.env_params:
            self.env = DefaultEnv(**self.env_params)
            self.actions = 2
            self.observations = (2,)
            self.gamma = 1.0
        elif self.env_params['type'] == 'Daily_ContextBandit':
            self.env = Daily_ContextBandit(**self.env_params)
            self.actions = 2
            self.observations = (1,)
            self.gamma = 0.0
        elif self.env_params['type'] == 'Daily_Bandit':
            self.env = Daily_Bandit(**self.env_params)
            self.actions = 2
            self.observations = (1,)
            self.gamma = 0.0
        elif self.env_params['type'] == 'Daily_ESARSA_TOD':
            self.env = Daily_ESARSA_TOD(**self.env_params)
            self.actions = 2
            self.observations = (1,)
            self.gamma = 1.0
        else:
            raise ValueError(f"{self.env_params['type']} is an invalid argument for SimplePlantSimulator type.")
