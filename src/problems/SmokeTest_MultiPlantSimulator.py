from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import SmokeTest_MultiPlantSimulator as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging 

class SmokeTest_MultiPlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        
        self.env = Env(**self.env_params)
        self.actions = 2
        self.observations = (4,) 
        self.gamma = 0.99