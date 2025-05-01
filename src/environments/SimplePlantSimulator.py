import os
import numpy as np
import pandas as pd
from RlGlue.environment import BaseEnvironment
from utils.functions import PiecewiseLinear

class SimplePlantSimulator(BaseEnvironment):
    '''
    Simulate a tray of plants under the same lighting agent.
    State = [linear time-of-day, 
             average area, 
             percentage change in average area over 1 step (for tracking plant motion)
             ] 
    Action = [moonlight, low, standard, high]
    Reward = raw change in average area over 1 step (includes large overnight growth)
    '''
    def __init__(self, last_day=14, **kwargs):
        self.state_dim = (3,)
        self.current_state = np.empty(3)
        self.action_dim = 4
        self.actions = [0, 1, 2, 3]

        self.observed_areas = []        # stores a list of lists of daytime observed areas in pixels. i.e. self.observed_areas[-1] contains the latest areas of individual plants

        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.num_steps = 0
        self.last_day = last_day        # horizon in units of days
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today
        self.num_plants = None          # num_plants will be the total number of plants in the data

        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_areas, self.projection_factors, self.terminal_step = self.analyze_area_data()

        self.gamma = 1.0
    
    def get_observation(self):
        # Compute observed areas by projecting actual areas
        self.observed_areas.append(np.array([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] for i in range(self.num_plants)]))        
        
        # Compute the average of the current observed areas
        new_area = np.mean(self.observed_areas[-1])

        # Compute plant motion, defined as percent change in average area per step
        if self.num_steps == 0: 
            plant_motion = 0
        elif self.num_steps % self.steps_per_day == 0:   # calculate % growth per overnight step, assuming constant % growth  
            old_area = np.mean(self.observed_areas[-2]) 
            total_growth = new_area / old_area - 1
            plant_motion = (total_growth + 1)**(1 / self.steps_per_night) - 1
        else: 
            old_area = np.mean(self.observed_areas[-2]) 
            plant_motion = new_area / old_area - 1
        
        observation = np.hstack([self.time_of_day(), 
                                 self.normalize(new_area, l=0, u=8000),
                                 self.normalize(plant_motion, l=-0.05, u=0.05)])
            
        return observation

    def start(self):
        self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]   # Make a copy because actual_areas will be modified at each step

        self.frozen_time_today = 0
        self.time = 0
        self.num_steps = 0
        self.observed_areas = []

        self.current_state = self.get_observation()

        return self.current_state

    def step(self, action):
        # Modify the interpolated actual_areas according to the action
        for pwl in self.actual_areas:
            pwl.insert_plateau(self.time, self.time + self.frozen_time(action))
        self.frozen_time_today += self.frozen_time(action)

        # Keep track of time 
        self.time += 1   # must occur after the above action effect
        self.num_steps += 1

        # Overnight behaviors
        if self.num_steps % self.steps_per_day == 0:  # if transitioning into the next day
            # freeze growth at night by the same amount as daytime today
            for pwl in self.actual_areas:
                pwl.insert_plateau(self.time, self.time + self.frozen_time_today)
            self.time += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        self.current_state = self.get_observation()
        
        self.reward = self.reward_function()

        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:
            return self.reward, self.current_state, False, self.get_info()

    def reward_function(self):
        new_area = np.mean(self.observed_areas[-1])
        old_area = np.mean(self.observed_areas[-2])
        return self.normalize(new_area - old_area, 0, 500)   # normalize typical reward to [0, 1]

    def get_info(self):
        return {"gamma": self.gamma}

    def time_of_day(self):
        step_today = self.num_steps % self.steps_per_day
        return step_today / self.steps_per_day

    def normalize(self, x, l, u): 
        return (x - l) / (u - l)

    def frozen_time(self, action):
        # Amount of frozen time (in unit of time step), given action
        all_time = {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0}  
        return all_time[action]
        
    def analyze_area_data(self):    # Approximate the actual leaf sizes and the projection factor throughout the day
        PWL = []
        PF = []
        for _ in range(self.num_plants):
            ep_data = self.data[:, _]
            observed_area = np.reshape(ep_data, (-1, self.steps_per_day))  # reshape into different days
            max_indices = np.argmax(observed_area, axis=1)                 # index at the max value of each day

            # Compute a piecewise linear function that interpolates between time stamps at daily max values. Include night times in the function.
            max_time = []
            max_area = []
            for i in range(observed_area.shape[0]):
                max_time.append((i-1)*(self.steps_per_day+self.steps_per_night) + max_indices[i])  # Let time begins at the start of day 2
                max_area.append(ep_data[i*self.steps_per_day + max_indices[i]])
            pwl = PiecewiseLinear(max_time, max_area)

            # Number of remaining daytime time stamps (since we truncate the first and last day)
            terminal_step = min((observed_area.shape[0]-2)*self.steps_per_day - 1, self.last_day*self.steps_per_day)

            # Compute the actual area at all daytime time stamps
            actual_area_daytime = []
            for i in range(observed_area.shape[0]-2):
                x_values = np.arange(i*(self.steps_per_day+self.steps_per_night), i*(self.steps_per_day+self.steps_per_night)+self.steps_per_day)
                y_values = [pwl(x) for x in x_values]
                actual_area_daytime.append(y_values)
            actual_area_daytime = np.hstack(actual_area_daytime)

            # Compute projection factor
            truncated_data = ep_data[self.steps_per_day:-self.steps_per_day]
            projection_factor = truncated_data/actual_area_daytime

            PWL.append(pwl)
            PF.append(projection_factor)

        return PWL, PF, terminal_step

    def load_area_data(self):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        df = pd.read_csv(data_path).sort_values(by='timestamp')

        # Filter out any plants that have sensor reading errors (area randomly goes to 0 at some timesteps)
        # TODO impute missing values by repeating previous values
        df = df.loc[:, ~(df == 0).any()]

        # Number of time steps per day
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timestamps_per_day = df['timestamp'].dt.date.value_counts()
        if timestamps_per_day.nunique() != 1:
            raise ValueError(f"Inconsistent timestamps per day: {timestamps_per_day.to_dict()}")
        steps_per_day = timestamps_per_day.iloc[0]

        # Number of time steps per night
        time_increment = df['timestamp'].diff().mode()[0]
        night_duration = df.groupby(df['timestamp'].dt.date)['timestamp'].first().shift(-1) - df.groupby(df['timestamp'].dt.date)['timestamp'].last()
        steps_per_night = int((night_duration.mode()[0] / time_increment) - 1)

        # The second when the first day starts
        first_second = pd.to_datetime(df['timestamp'].iloc[0]).time().hour * 3600 + pd.to_datetime(df['timestamp'].iloc[0]).time().minute * 60

        # Observed areas of plants (in unit of pixels)
        plant_area_data = np.array(df.drop(columns=['timestamp']))

        # Check if we have enough data for the requested last_day
        assert plant_area_data.shape[0] / steps_per_day - 2 >= self.last_day, f'The requested last_day exceeds available plant data, which has {plant_area_data.shape[0] / steps_per_day - 2} days.'

        self.num_plants = plant_area_data.shape[1]

        return plant_area_data, steps_per_day, steps_per_night, time_increment.total_seconds(), first_second
    
class BanditEnv(SimplePlantSimulator):
    '''
    Only one state. Gamma = 0. Overnight reward is set to 0.
    '''
    def __init__(self, last_day=14, **kwargs):
        super().__init__(last_day, **kwargs)     
        self.state_dim = (1,)
        self.current_state = np.empty(1)
        self.gamma = 0.0

    def get_observation(self):      
        super().get_observation()
        return np.array([1])

    def reward_function(self):
        if self.num_steps % self.steps_per_day == 0: 
            return 0 
        else: 
            new_area = np.mean(self.observed_areas[-1])
            old_area = np.mean(self.observed_areas[-2])
            return self.normalize(new_area - old_area, 0, 500)   # normalize typical reward to [0, 1]