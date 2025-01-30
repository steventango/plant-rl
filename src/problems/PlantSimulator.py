from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import PlantSimulator as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class PlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env()  # default is one plant and two actions (off and on)
        self.actions = 2
        self.observations = (2,)
        self.gamma = 0.99