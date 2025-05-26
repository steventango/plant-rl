from PyExpUtils.collection.Collector import Collector
from environments.Gym import Gym

from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class CartPole(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Gym(name="CartPole-v1", seed=self.seed, sutton_barto_reward=True)
        self.actions = 2

        # encode the observation ranges for this problem
        # useful for tile-coding
        self.rep_params['input_ranges'] = [
            [-4.8, 4.6],
            [-6, 6],
            [-.2095, .2095],
            [-2, 2]
        ]

        self.observations = (4,)
        self.gamma = 0.999
