import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import iqm
from utils.metrics import UnbiasedExponentialMovingAverage as uema


class CVPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, zone: int, total_steps: int = 40320):
        super().__init__(zone)
        self.total_steps = total_steps

        self.history = uema(alpha=0.01)   # growth rate = trace of (% change in area over 1 time step)
        self.current_state = np.empty(2)
 
    def get_observation(self):
        time, _, plant_stats = super().get_observation()

        if len(self.observed_areas) >= 2:
            old_area = iqm(self.observed_areas[-2], self.q)
            new_area = iqm(self.observed_areas[-1], self.q)
            self.history.update(self.percent_change(old_area, new_area))

        time_of_day = self.transform_time_linear(time)  # TODO: DQN needs sin/cos time, ESARSA needs linear
        
        observation = np.hstack([time_of_day, 
                                 self.normalize(self.history.compute())])
        
        return observation
    
    def start(self):
        self.history.reset() 
        self.observed_areas = []
        self.current_state = self.get_observation()
        return self.current_state

    def step_two(self):
        self.current_state = self.get_observation()
        self.reward = self.reward_function()

        return self.reward, self.current_state, False, self.get_info()

    def reward_function(self):   # reward = last state input = smooth change in area
        return self.current_state[-1]

    def percent_change(self, old, new):   # symmetric percentage change
        return 2 * (new - old) / (new + old)
    
    def normalize(self, x, l=0.0005, u=0.0025):  
        return (x - l) / (u - l)
    
    def transform_time_sine(self, time, total=86400.0):
        return np.array([np.sin(2 * np.pi * time / total), np.cos(2 * np.pi * time / total)])

    def transform_time_linear(self, time, total=86400.0):
        return time / total
        
