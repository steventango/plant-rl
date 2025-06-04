from PyExpUtils.collection.Collector import Collector
from environments.CleanPlantSimulator import CleanPlantSimulator, CleanPlantSimulator_Daily
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class SimplePlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.actions = 2
        self.observations = (3,)
        self.gamma = 1.0
        if self.env_params['type'] == 'default':
            self.env = CleanPlantSimulator(**self.env_params)
        elif self.env_params['type'] == 'daily':
            self.env = CleanPlantSimulator_Daily(**self.env_params)
        else:
            raise ValueError(f"Invalid argument. Expected one of {{'default', 'daily'}}, but got: {self.env_params['type']}")