from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import PlantSimulator as PlantSimulatorEnv
from environments.PlantSimulator import PlantSimulatorLowHigh as PlantSimulatorEnvLowHigh
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging 

class PlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        if self.env_params['type'] == 'default':
            self.env = PlantSimulatorEnv(**self.env_params)
            self.actions = 4
        elif self.env_params['type'] == 'low_high':
            self.env = PlantSimulatorEnvLowHigh(**self.env_params)
            self.actions = 2
        else:
            raise ValueError(f"Invalid argument. Expected one of {{'default', 'low_high'}}, but got: {self.env_params['type']}")
        
        self.observations = (6,) 
        self.gamma = 1.0