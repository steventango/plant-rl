import os
import numpy as np
import pandas as pd
from RlGlue.environment import BaseEnvironment

class PlantSimulator(BaseEnvironment):
    def __init__(self):
        self.gamma = 0.99
        self.state_dim = (2,)    
        self.current_state = np.empty(2)
        self.action_dim = 2      
        self.actions = [0, 1]             # [light off, light on]
        self.deduct_growth = [1.0, 0.0]   # 100% means no growth. 0% means growing optimally as in the hisotric data.
        self.num_steps = 0                # time step counter
        self.steps_per_day = 72           # number of time steps per day 
        
        # Load historic plant area data 
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        data = pd.read_csv(data_path).sort_values(by='timestamp')
        self.data = np.array(data.iloc[:, 5])  # Use one plant for now (plant 5 is good)

        # Approximate the actual leaf sizes and the projection factor throughout the day
        self.original_actual_area, self.daily_projection_factor = self.analyze_area_data()

        # Make a copy because actual_area will be modified at each step
        self.actual_area = np.copy(self.original_actual_area)  

    def start(self):
        self.num_steps = 0
        self.actual_area = np.copy(self.original_actual_area)
        self.current_state = np.array([0, self.actual_area[0]*self.daily_projection_factor[0]])
        return self.current_state

    def step(self, action): 
        self.num_steps += 1
        
        # To simulate growth reduction due to suboptimal lighting (assuming historic data came from optimal lighting)
        # reduce the leaf's actual_area in all subsequent time steps
        dA = self.actual_area[self.num_steps] - self.actual_area[self.num_steps - 1]   # growth in optimal lighting
        dA_deduction = self.deduct_growth[action]*dA  # deduction from dA
        self.actual_area -= dA_deduction   # affects all time steps, but the past data don't matter anyways

        # State := concatenate( time of day, observed leaf area )
        clock = self.num_steps % self.steps_per_day
        next_state = np.array([clock, self.actual_area[self.num_steps]*self.daily_projection_factor[clock]])
        
        self.reward = self.compute_reward(clock, next_state)

        self.current_state = next_state

        if self.num_steps == len(self.actual_area) - 1:    # ternimal state at the end of usable data 
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
            
    def compute_reward(self, clock, next_state):   # ignore overnight growth
        if clock == 0:
            return 0
        else:
            return next_state[1] - self.current_state[1]
    
    def get_info(self):
        return {"gamma": self.gamma}
    
    def analyze_area_data(self): 
        observed_area = np.reshape(self.data, (-1, self.steps_per_day))  # reshape into different days
        max_indices = np.argmax(observed_area, axis=1)        # index at the max value of each day
        
        actual_area = np.copy(self.data)
        for i in range(observed_area.shape[0]-1):
            min_id = i*self.steps_per_day + max_indices[i]
            max_id = (i+1)*self.steps_per_day + max_indices[i+1]
            # Assume the largest observed area in each day IS the actual area at that moment
            # Interpolate (linearly) between largest observed areas
            actual_area[min_id:max_id] = np.interp(np.arange(min_id, max_id), [min_id, max_id], [self.data[min_id],self.data[max_id]])
        
        # Ratio between observed and actual area 
        projection_factor = np.reshape(self.data / actual_area, (-1, self.steps_per_day))
        # Checked that this ratio has a characteristic shape, independent of the day, so we take the average
        mean_projection_factor = np.mean(projection_factor, axis=0)

        # Truncate first and last days, during which actual_area was not interpolated
        actual_area = actual_area[self.steps_per_day:-self.steps_per_day]

        # Normalize by the actual area on day 1
        actual_area = actual_area / actual_area[0]
        
        return actual_area, mean_projection_factor



