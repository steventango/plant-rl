from PyExpUtils.collection.Collector import Collector
from environments.PlantSimulator import PlantSimulator as PlantSimulatorEnv
from environments.PlantSimulator import PlantSimulatorLowHigh as PlantSimulatorEnvLowHigh
from environments.PlantSimulator import PlantSimulator_Only1Time_EMAReward
from environments.PlantSimulator import PlantSimulator_Only1Time
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

import logging 

class PlantSimulator(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        if self.env_params['type'] == 'default':
            self.env = PlantSimulatorEnv(**self.env_params)
            self.actions = 4
            self.observations = (6,) 
        elif self.env_params['type'] == 'only1time':
            self.env = PlantSimulator_Only1Time(**self.env_params)
            self.actions = 4
            self.observations = (3,)
        elif self.env_params['type'] == 'only1time_emareward':
            self.env = PlantSimulator_Only1Time_EMAReward(**self.env_params)
            self.actions = 4
            self.observations = (3,)
        elif self.env_params['type'] == 'low_high':
            self.env = PlantSimulatorEnvLowHigh(**self.env_params)
            self.actions = 2
            self.observations = (6,) 
        else:
            raise ValueError(f"Invalid argument. Expected one of {{'default', 'only1time', 'low_high'}}, but got: {self.env_params['type']}")
        
        self.gamma = 1.0