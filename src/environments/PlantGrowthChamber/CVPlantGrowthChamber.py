import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import iqm
from utils.metrics import UnbiasedExponentialMovingAverage as uema


class CVPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, zone: int, total_steps: int = 40320):
        super().__init__(zone)
        self.total_steps = total_steps

        self.history = uema(alpha=0.01)   # history of change in average observed area over 1 time step (in units of mm^2)
 
    def get_observation(self):
        time, _, plant_stats = super().get_observation()

        if len(self.observed_areas) >= 2:
            self.history.update(iqm(self.observed_areas[-1], self.q) - iqm(self.observed_areas[-2], self.q))
        
        time_of_day = self.transform_time_linear(time)
        
        observation = np.hstack([time_of_day,
                                 self.normalize(iqm(self.observed_areas[-1], self.q)),
                                 self.normalize(self.history.compute(), l=-0.41, u=2.48)])
        
        return observation
    
    def start(self):
        self.history.reset() 
        self.observed_areas = []
        observation = self.get_observation()
        return observation
    
    def reward_function(self):   # reward = last state input = smooth change in area
        return self.normalize(self.history.compute().item(), l=-0.41, u=2.48)

    def transform_time_sine(self, time, total=86400.0):
        return np.array([np.sin(2 * np.pi * time / total), np.cos(2 * np.pi * time / total)])

    def transform_time_linear(self, time, total=86400.0):
        return time / total
