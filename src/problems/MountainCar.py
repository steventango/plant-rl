from PyExpUtils.collection.Collector import Collector  # type: ignore
from PyRlEnvs.domains.MountainCar import GymMountainCar

from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class Env(GymMountainCar):
    def start(self):  # type: ignore
        return super().start(), {}


class MountainCar(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Env(seed=self.seed)
        self.actions = 3

        # encode the observation ranges for this problem
        # useful for tile-coding
        self.rep_params["input_ranges"] = [
            [-1.2, 0.5],
            [-0.07, 0.07],
        ]

        self.observations = (2,)
        self.gamma = 0.99
