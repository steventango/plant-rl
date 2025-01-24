import os
import numpy as np
import pandas as pd
from RlGlue.environment import BaseEnvironment

class PlantSimulator(BaseEnvironment):
    def __init__(self, plant_id=5):
        self.state_dim = (2,)    
        self.current_state = np.empty(2)
        self.action_dim = 2      
        self.actions = [0, 1]             # [light off, light on]
        self.deduct_growth = [1.0, 0.0]   # growth deduction due to action. 100% means no growth. 0% means growing optimally as in the hisotric data.
        
        self.data, self.steps_per_day = self.load_area_data(plant_id)
        self.original_actual_area, self.daily_projection_factor = self.analyze_area_data()
        self.actual_area = np.copy(self.original_actual_area)   # Make a copy because actual_area will be modified at each step
        self.observation = []   # store a list of observed areas

        self.gamma = 0.99
        self.num_steps = 0

    def start(self):
        self.num_steps = 0
        self.actual_area = np.copy(self.original_actual_area)
        self.current_state = np.array([0, self.actual_area[0]*self.daily_projection_factor[0]])
        self.observation.append(self.current_state[1])
        return self.current_state

    def step(self, action): 
        self.num_steps += 1
        
        # Modify actual_area in all subsequent time steps due to lighting choice
        dA = self.actual_area[self.num_steps] - self.actual_area[self.num_steps - 1]  # growth in optimal lighting
        dA_deduction = self.deduct_growth[action]*dA    # deduction from dA
        self.actual_area -= dA_deduction   # affects all time steps (past data aren't used again anyways)
        
        # Compute observed area by projecting actual area
        clock = self.num_steps % self.steps_per_day
        self.observation.append(self.actual_area[self.num_steps]*self.daily_projection_factor[clock])

        # Compute reward
        self.reward = self.reward_function()

        # Define state as concatenate( time of day, observed leaf area )
        self.current_state = np.array([clock, self.observation[-1]])

        if self.num_steps == len(self.actual_area) - 1:    # ternimal state when data runs out
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
        
    def reward_function(self):  
        ''' Reward = next observed area - current observed area '''
        if self.num_steps % self.steps_per_day == 0:   # ignore overnight growth
            return 0
        else:
            return self.observation[-1] - self.observation[-2]
        
    def reward_function1(self):  
        ''' Reward = next observed area - observed area exactly a day prior. Only available on day 2 '''
        if self.num_steps >= self.steps_per_day: 
            return (self.observation[-1] - self.observation[-1-self.steps_per_day])
        else: 
            return 0
    
    def get_info(self):
        return {"gamma": self.gamma}
    
    def analyze_area_data(self):    
        ''' Approximate the actual leaf sizes and the projection factor throughout the day '''

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
    
    def load_area_data(self, plant_id):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        df = pd.read_csv(data_path).sort_values(by='timestamp')

        # Compute number of time steps per day 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timestamps_per_day = df['timestamp'].dt.date.value_counts()
        if timestamps_per_day.nunique() != 1:
            raise ValueError(f"Inconsistent timestamps per day: {timestamps_per_day.to_dict()}")
        
        return np.array(df.iloc[:, plant_id]), timestamps_per_day.iloc[0]