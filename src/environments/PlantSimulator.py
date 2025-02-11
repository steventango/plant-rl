import os
from math import sin, cos, pi
import random

import numpy as np
import pandas as pd

from RlGlue.environment import BaseEnvironment
from utils.functions import PiecewiseLinear
from math import sin, cos, pi

class PlantSimulator(BaseEnvironment):
    def __init__(self, plant_id=[1], actions=[0, 1], action_effects=[1.0, 0.0]):
        self.state_dim = (5,)     
        self.current_state = np.empty(5)
        self.action_dim = 2      
        self.actions = actions               # default is [light off, light on]
        self.frozen_time = action_effects    # due to the agent's action, freeze plant for a percentage of the current time step 
        self.randomize_plant = randomize_plant
        self.plant_id = [random.randint(1, 64)] if randomize_plant else plant_id
        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data(plant_id)
        self.original_actual_area, self.projection_factor, self.terminal_step = self.analyze_area_data()
        self.actual_area = self.original_actual_area.copy()   # Make a copy because actual_area will be modified at each step
        self.ob = []                    # store a list of observed areas
        self.smooth_ob = []             # store a list of moving-averaged observed areas
        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today

        self.gamma = 0.99
        self.num_steps = 0
        self.n_step = 1 # Sets lag for determining change in area used in reward function (72 = 1 day)

    def start(self):
        self.num_steps = 0
        self.time = 0
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds

        self.frozen_time_today = 0
        self.actual_area = self.original_actual_area.copy()

        self.ob.append(self.actual_area(self.time)*self.projection_factor[self.num_steps])
        self.smooth_ob.append(self.ob[-1])

        # State = Concatenate(sine time, normalized time since beginning, normalized observed area, normalized moving-averaged observed area)
        self.current_state = np.hstack([self.sine_time(clock), self.num_steps/self.terminal_step, self.normalize([self.ob[-1], 0])])
        return self.current_state

    def step(self, action): 
        # Modify the interpolated actual_area according to the action
        self.actual_area.insert_plateau(self.time, self.time + self.frozen_time[action])
        self.frozen_time_today += self.frozen_time[action]

        # Keep track of time
        self.num_steps += 1
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds
        self.time += 1

        # If transitioning into the next day, freeze growth at night by the same amount as daytime today
        if self.num_steps % self.steps_per_day == 0: 
            self.actual_area.insert_plateau(self.time, self.time + self.frozen_time_today)
            self.time += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        # Compute observed area by projecting actual area
        self.ob.append(self.actual_area(self.time)*self.projection_factor[self.num_steps])
        self.smooth_ob.append(self.moving_average(self.ob[-1]))

        # Define state
        if self.num_steps >= self.n_step: 
            self.current_state = np.hstack([self.sine_time(clock), self.num_steps/self.terminal_step, self.normalize([self.ob[-1], self.ob[-1-self.n_step]])])
        else: 
            self.current_state = np.hstack([self.sine_time(clock), self.num_steps/self.terminal_step, self.normalize([self.ob[-1], 0])])

        # Compute reward
        self.reward = self.reward_function_n_step(n_step=self.n_step)

        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
    
    def get_info(self):
        return {"gamma": self.gamma}
        
    def reward_function_n_step(self, n_step=1):
        if self.num_steps >= n_step: 
            return (self.ob[-1] - self.ob[-1 - n_step]) / self.ob[-1 - n_step]
        else: 
            return 0
        
    def analyze_area_data(self):    
        ''' Approximate the actual leaf sizes and the projection factor throughout the day '''

        observed_area = np.reshape(self.data, (-1, self.steps_per_day))  # reshape into different days
        max_indices = np.argmax(observed_area, axis=1)        # index at the max value of each day
        
        # Compute a piecewise linear function that interpolates between time stamps at daily max values. Include night times in the function.
        max_time = []
        max_area = []
        for i in range(observed_area.shape[0]):
            max_time.append((i-1)*(self.steps_per_day+self.steps_per_night) + max_indices[i])  # Let time begins at the start of day 2
            max_area.append(self.data[i*self.steps_per_day + max_indices[i]])
        pwl = PiecewiseLinear(max_time, max_area)

        # Number of remaining daytime time stamps (since we truncate the first and last day)
        terminal_step = (observed_area.shape[0]-2)*self.steps_per_day - 1

        # Compute the actual area at all daytime time stamps
        actual_area_daytime = []
        for i in range(observed_area.shape[0]-2):
            x_values = np.arange(i*(self.steps_per_day+self.steps_per_night), i*(self.steps_per_day+self.steps_per_night)+self.steps_per_day)
            y_values = [pwl(x) for x in x_values]
            actual_area_daytime.append(y_values)
        actual_area_daytime = np.hstack(actual_area_daytime)

        # Compute projection factor
        truncated_data = self.data[self.steps_per_day:-self.steps_per_day]
        projection_factor = truncated_data/actual_area_daytime
        
        return pwl, projection_factor, terminal_step
    
    def load_area_data(self, plant_id):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        df = pd.read_csv(data_path).sort_values(by='timestamp')
        
        # The second when  the first day starts 
        first_second = pd.to_datetime('2024-02-10 09:00').time().hour * 3600 + pd.to_datetime('2024-02-10 09:00').time().minute * 60

        # Number of time steps per day 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timestamps_per_day = df['timestamp'].dt.date.value_counts()
        if timestamps_per_day.nunique() != 1:
            raise ValueError(f"Inconsistent timestamps per day: {timestamps_per_day.to_dict()}")
        steps_per_day = timestamps_per_day.iloc[0]
        
        # Number of time steps per night
        time_increment = df['timestamp'].diff().mode()[0]
        night_duration = df.groupby(df['timestamp'].dt.date)['timestamp'].first().shift(-1) - df.groupby(df['timestamp'].dt.date)['timestamp'].last()
        print(df)
        # Averaged observed plant area (in unit of pixels)
        plant_area_data = np.array(df.iloc[:, plant_id].mean(axis=1))

        return plant_area_data, steps_per_day, steps_per_night, time_increment.total_seconds(), first_second
    
    def normalize(self, x):   # normalize observation to between 0 and 1
        u = 30000   # max historic area of one plant (in pixels)
        l = 0
        if isinstance(x, list):
            return [(val - l) / (u - l) for val in x]
        return (x - l) / (u - l)
    
    def sine_time(self, t):
        # Return sine & cosine times, normalized to between 0 and 1
        return [(sin(2*pi*t/86400)+1)/2, (cos(2*pi*t/86400)+1)/2]
    
    def moving_average(self, x, trace_decay_rate = 0.99):
        return trace_decay_rate *self.smooth_ob[-1] + (1-trace_decay_rate)*x

