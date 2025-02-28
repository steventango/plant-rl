import numpy as np
from environments.Gridworld.Gridworld import Gridworld as Env
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class Gridworld(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Env()
        self.actions = self.env.env.action_space.n

        self.observations = (self.env.env.observation_space.n,)
        self.gamma = 0.99
        
        
        self.rep_params['input_ranges'] = [
            [0,1],
            [0,1],
            [0,1],
            [0,1],
        ]