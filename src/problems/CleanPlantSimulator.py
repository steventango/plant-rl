from PyExpUtils.collection.Collector import Collector

from environments.CleanPlantSimulator import CleanPlantSimulator
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class SimplePlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.actions = 2
        self.observations = (4,)
        self.gamma = 1.0
        self.env = CleanPlantSimulator(**self.env_params)
