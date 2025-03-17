from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import PlantSimulator 
from environments.PlantSimulator import PlantSimulatorLowHigh
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging 

class PlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        if self.env_params['type'] == 'default':
            self.env = PlantSimulator(**self.env_params)
            self.actions = 4
        elif self.env_params['type'] == 'low_high':
            self.env = PlantSimulatorLowHigh(**self.env_params)
            self.actions = 2
        
        self.observations = (6,) 
        self.gamma = 1.0