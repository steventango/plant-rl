from environments.CliffWalking import CliffWalking as Env
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class CliffWalking(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Env()
        self.actions = 4

        self.observations = (self.env.env.observation_space.n,)
        self.gamma = 1
