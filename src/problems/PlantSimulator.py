from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import PlantSimulator as PlantSimulatorOffLowHigh
from environments.PlantSimulator import PlantSimulatorLowHigh
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging 

class PlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        if self.env_params['type'] == 'off_low_high':
            self.env = PlantSimulatorOffLowHigh(**self.env_params)
            self.actions = 3
        elif self.env_params['type'] == 'low_high':
            self.env = PlantSimulatorLowHigh(**self.env_params)
            self.actions = 2
        
        self.observations = (6,) 
        self.gamma = 1.0