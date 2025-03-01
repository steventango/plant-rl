import os
from math import sin, cos, pi

import numpy as np
import pandas as pd

from RlGlue.environment import BaseEnvironment
from utils.functions import PiecewiseLinear

class PlantSimulator(BaseEnvironment):
    def __init__(self, seed=0, plant_id=0, random_plant=False, lag=None, actions=[0, 1], action_effects=[1.0, 0.0]):
        self.seed = seed
        self.rand_state = np.random.RandomState(seed)
        self.state_dim = (5,)     
        self.current_state = np.empty(5)
        self.action_dim = 2      
        self.actions = actions               # default is [light off, light on]
        self.frozen_time = action_effects    # due to the agent's action, freeze plant for a percentage of the current time step 
        self.use_random_plant = random_plant
        self.plant_id = plant_id

        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_area, self.projection_factor, self.terminal_step = self.analyze_area_data()
        self.observed_area = []                    # store a list of observed areas
        #self.smooth_ob = []             # store a list of moving-averaged observed areas
        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today

        self.gamma = 0.99
        self.num_steps = 0

        self.lag = lag if lag is not None else self.steps_per_day # Sets lag for determining change in area used in reward function, default is 1 day

    def start(self):
        self.num_steps = 0
        self.time = 0
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds

        self.frozen_time_today = 0

        if self.use_random_plant:
            self.plant_id = self.rand_state.randint(0, self.data.shape[1])
            self.original_actual_area, self.projection_factor, self.terminal_step = self.analyze_area_data()

        self.actual_area = self.original_actual_area.copy()   # Make a copy because actual_area will be modified at each step

        self.observed_area.append(self.actual_area(self.time)*self.projection_factor[self.num_steps])
        #self.smooth_ob.append(self.ob[-1])

        # State = Concatenate(sine time, normalized time since beginning, normalized observed area, normalized moving-averaged observed area)
        self.current_state = np.hstack([self.sine_time(clock), self.num_steps/self.terminal_step, self.normalize([self.observed_area[-1], 0])])
        return self.current_state

    def step(self, action): 
        #TODO Change so that observed area is 0 in darkness (or maybe a special indicator?) But the reward 
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
        self.observed_area.append(self.actual_area(self.time)*self.projection_factor[self.num_steps])
        #self.smooth_ob.append(self.moving_average(self.ob[-1]))

        # Define state
        if self.num_steps >= self.lag: 
            self.current_state = np.hstack([self.sine_time(clock), self.num_steps/self.terminal_step, self.normalize([self.observed_area[-1], self.observed_area[-1-self.lag]])])
        else: 
            self.current_state = np.hstack([self.sine_time(clock), self.num_steps/self.terminal_step, self.normalize([self.observed_area[-1], 0])])

        # Compute reward
        self.reward = self.reward_function_lag(lag=self.lag)

        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
    
    def get_info(self):
        return {"gamma": self.gamma}
        
    def reward_function_lag(self, lag=1):
        if self.num_steps >= lag: 
            return (self.observed_area[-1] - self.observed_area[-1 - lag]) / self.observed_area[-1 - lag]
        else: 
            return 0
        
    def analyze_area_data(self):    
        ''' Approximate the actual leaf sizes and the projection factor throughout the day '''
        ep_data = self.data[:, self.plant_id] # Get the data for the plant selected for the current episode
        observed_area = np.reshape(ep_data, (-1, self.steps_per_day))  # reshape into different days
        max_indices = np.argmax(observed_area, axis=1)        # index at the max value of each day
        
        # Compute a piecewise linear function that interpolates between time stamps at daily max values. Include night times in the function.
        max_time = []
        max_area = []
        for i in range(observed_area.shape[0]):
            max_time.append((i-1)*(self.steps_per_day+self.steps_per_night) + max_indices[i])  # Let time begins at the start of day 2
            max_area.append(ep_data[i*self.steps_per_day + max_indices[i]])
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
        truncated_data = ep_data[self.steps_per_day:-self.steps_per_day]
        '''
        TODO When any of the areas from the dataset are 0, the projection factor goes to 0. 
        So multiplying the actual_area by the projection factor to get the observed area in step 
        results in a value of 0. Then when we calculate the % change for the reward we get a divide by 0 
        error. For now, we can just use plants that don't have any 0 entries, but in the future we 
        might want to include random errors in sensor readings like these when training the model so 
        we can ensure the algo/hypers we deploy are robust to sensor errors (this has been mentioned previously by Adam as well). 
        '''
        projection_factor = truncated_data/actual_area_daytime
        
        return pwl, projection_factor, terminal_step
    
    def load_area_data(self):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        df = pd.read_csv(data_path).sort_values(by='timestamp')
        # Filter out any plants that have sensor reading errors (area randomly goes to 0 at some timesteps)
        df = df.loc[:, ~(df == 0).any()]
        # The second when  the first day starts 
        first_second = pd.to_datetime(df['timestamp'].iloc[0]).time().hour * 3600 + pd.to_datetime(df['timestamp'].iloc[0]).time().minute * 60

        # Number of time steps per day 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timestamps_per_day = df['timestamp'].dt.date.value_counts()
        if timestamps_per_day.nunique() != 1:
            raise ValueError(f"Inconsistent timestamps per day: {timestamps_per_day.to_dict()}")
        steps_per_day = timestamps_per_day.iloc[0]
        
        # Number of time steps per night
        time_increment = df['timestamp'].diff().mode()[0]
        night_duration = df.groupby(df['timestamp'].dt.date)['timestamp'].first().shift(-1) - df.groupby(df['timestamp'].dt.date)['timestamp'].last()
        steps_per_night = int((night_duration.mode()[0] / time_increment)-1)

        # Averaged observed plant area (in unit of pixels)
        plant_area_data = np.array(df.drop(columns=['timestamp']))
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
    '''
    def moving_average(self, x, trace_decay_rate = 0.99):
        return trace_decay_rate *self.smooth_ob[-1] + (1-trace_decay_rate)*x
    '''

class MultiPlantSimulator(BaseEnvironment):  
    ''' 
    Simulate a tray of plants under the same lighting agent.
    State = (time of day, average lagged area, average area)
    '''
    def __init__(self, num_plants=32, lag=1, stride=1, actions=[0, 1], action_effects=[1.0, 0.0], one_hot_time = False):
        self.state_dim = (one_hot_time*22 + 4,) # Time encoding has len 24 if one-hot else len 2 if sin/cos, plus 2 area features 
        self.current_state = np.empty(self.state_dim)
        self.action_dim = len(actions)      
        self.actions = actions               # default is [light off, light on]
        self.frozen_time = action_effects    # due to the agent's action, freeze plant for a percentage of the current time step 
        self.one_hot_time = one_hot_time

        self.num_plants = num_plants
        self.stride = stride                 # env time step = stride * time step in plant data
        self.lag = lag                       # lag for change in area used in reward function; default is 1 time step

        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_areas, self.projection_factors, self.terminal_step = self.analyze_area_data()

        self.observed_areas = []        # store a list of observed areas
        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.num_steps = 0
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today

        self.gamma = 0.99

    def start(self):
        self.frozen_time_today = 0
        self.time = 0
        self.num_steps = 0
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds

        self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]   # Make a copy because actual_areas will be modified at each step

        self.observed_areas.append([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] 
                                    for i in range(self.num_plants)])

        self.current_state = np.hstack([self.sine_time(clock), 
                                        self.normalize(1),   # fill in missing value(s) with small value(s) close to zero
                                        self.normalize(np.mean(self.observed_areas[-1]))])

        return self.current_state

    def step(self, action): 
        # Modify the interpolated actual_areas according to the action
        for pwl in self.actual_areas: 
            pwl.insert_plateau(self.time, self.time + self.frozen_time[action])
        self.frozen_time_today += self.frozen_time[action]

        # Keep track of time
        self.time += 1
        self.num_steps += 1
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds

        # If transitioning into the next day, freeze growth at night by the same amount as daytime today
        if self.num_steps % self.steps_per_day == 0: 
            for pwl in self.actual_areas:
                pwl.insert_plateau(self.time, self.time + self.frozen_time_today)
            self.time += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        # Compute observed areas by projecting actual areas
        self.observed_areas.append([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] 
                                    for i in range(self.num_plants)])
        if self.one_hot_time:
            time_obs = np.zeros(24)
            time_obs[clock // 3600] = 1 # Set the position for the current hour to 1
        else:
            time_obs = self.sine_time(clock)
        # Set state
        if self.num_steps >= self.lag: 
            self.current_state = np.hstack([time_obs, 
                                            self.normalize(np.mean(self.observed_areas[-1-self.lag])),   
                                            self.normalize(np.mean(self.observed_areas[-1]))])
        else: 
            self.current_state = np.hstack([time_obs, 
                                            self.normalize(1),   
                                            self.normalize(np.mean(self.observed_areas[-1]))])

        #self.reward = self.reward_function_raw()
        self.reward = self.reward_function()

        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
    
    def reward_function(self):
        if self.num_steps >= self.lag: 
            new = self.observed_areas[-1]
            old = self.observed_areas[-1-self.lag]
            return np.mean(new) / np.mean(old) - 1
        else: 
            return 0

    def reward_function_raw(self):
        if self.num_steps >= self.lag: 
            new = self.normalize(np.mean(self.observed_areas[-1]))
            old = self.normalize(np.mean(self.observed_areas[-1-self.lag]))
            return (new - old) / 0.08
        else: 
            return 0

    def get_info(self):
        return {"gamma": self.gamma}
        
    def analyze_area_data(self):    
        ''' 
        Approximate the actual leaf sizes and the projection factor throughout the day 
        '''
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
            terminal_step = (observed_area.shape[0]-2)*self.steps_per_day - 1

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

        # Optionally set a larger time step than the one given in plant data
        assert self.steps_per_day % self.stride == 0 & self.steps_per_night % self.stride == 0, f"stride must be a divisor of steps_per_day={self.steps_per_day} and of steps_per_night={self.steps_per_night}."
        PF = [pf[::self.stride] for pf in PF]
        PWL = [pwl.rescale_x(1 / self.stride) for pwl in PWL]
        self.steps_per_day /= self.stride
        self.steps_per_night /= self.stride
        self.interval *= self.stride
        terminal_step = int(terminal_step / self.stride)

        return PWL, PF, terminal_step
    
    def load_area_data(self):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        df = pd.read_csv(data_path).sort_values(by='timestamp')

        # Filter out any plants that have sensor reading errors (area randomly goes to 0 at some timesteps)
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
        steps_per_night = int((night_duration.mode()[0] / time_increment)-1)

        # The second when the first day starts 
        first_second = pd.to_datetime(df['timestamp'].iloc[0]).time().hour * 3600 + pd.to_datetime(df['timestamp'].iloc[0]).time().minute * 60

        # Observed areas of plants (in unit of pixels)
        plant_area_data = np.array(df.drop(columns=['timestamp']))
        assert plant_area_data.shape[1] >= self.num_plants, f"Please request fewer than {plant_area_data.shape[1]} plants."

        return plant_area_data[:,:self.num_plants], steps_per_day, steps_per_night, time_increment.total_seconds(), first_second
    
    def sine_time(self, t):
        # Return sine & cosine times, normalized to between 0 and 1
        return [(sin(2*pi*t/86400)+1)/2, (cos(2*pi*t/86400)+1)/2]
    
    def normalize(self, x):   # normalize observation to between 0 and 1
        u = 30000   # max historic area of one plant (in pixels)
        l = 0
        if isinstance(x, list):
            return [(val - l) / (u - l) for val in x]
        return (x - l) / (u - l)

class SmokeTest_MultiPlantSimulator(BaseEnvironment):  
    ''' 
    Simulate a tray of plants under the same lighting agent.
    State = (time of day, average lagged area, average area)
    Area is manually set to a small number when light is off
    Reward = average area
    '''
    def __init__(self, num_plants=32, lag=1, stride=1, actions=[0, 1], action_effects=[1.0, 0.0]):
        self.state_dim = (4,)   
        self.current_state = np.empty(4)
        self.action_dim = len(actions)      
        self.actions = actions               # default is [light off, light on]
        self.frozen_time = action_effects    # due to the agent's action, freeze plant for a percentage of the current time step 

        self.num_plants = num_plants
        self.stride = stride                 # env time step = stride * time step in plant data
        self.lag = lag                       # lag for change in area used in reward function; default is 1 time step

        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_areas, self.projection_factors, self.terminal_step = self.analyze_area_data()

        self.observed_areas = []        # store a list of observed areas
        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.num_steps = 0
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today

        self.gamma = 0.99

    def start(self):
        self.frozen_time_today = 0
        self.time = 0
        self.num_steps = 0
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds

        self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]   # Make a copy because actual_areas will be modified at each step

        self.observed_areas.append([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] 
                                    for i in range(self.num_plants)])

        self.current_state = np.hstack([self.sine_time(clock), 
                                        self.normalize(1),   # fill in missing value(s) with small value(s) close to zero
                                        self.normalize(np.mean(self.observed_areas[-1]))])

        return self.current_state

    def step(self, action): 
        # Modify the interpolated actual_areas according to the action
        for pwl in self.actual_areas: 
            pwl.insert_plateau(self.time, self.time + self.frozen_time[action])
        self.frozen_time_today += self.frozen_time[action]

        # Keep track of time
        self.time += 1
        self.num_steps += 1
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds

        # If transitioning into the next day, freeze growth at night by the same amount as daytime today
        if self.num_steps % self.steps_per_day == 0: 
            for pwl in self.actual_areas:
                pwl.insert_plateau(self.time, self.time + self.frozen_time_today)
            self.time += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        # Compute observed areas by projecting actual areas
        if action == 0: # if light is off, set areas to 1 pixel
            self.observed_areas.append([1 for i in range(self.num_plants)])
        else:
            self.observed_areas.append([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] 
                                        for i in range(self.num_plants)])


        # Set state
        if self.num_steps >= self.lag: 
            self.current_state = np.hstack([self.sine_time(clock), 
                                            self.normalize(np.mean(self.observed_areas[-1-self.lag])),   
                                            self.normalize(np.mean(self.observed_areas[-1]))])
        else: 
            self.current_state = np.hstack([self.sine_time(clock), 
                                            self.normalize(1),   
                                            self.normalize(np.mean(self.observed_areas[-1]))])

        self.reward = self.reward_function()

        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
    
    def reward_function(self):
        return self.normalize(np.mean(self.observed_areas[-1]))

    def get_info(self):
        return {"gamma": self.gamma}
        
    def analyze_area_data(self):    
        ''' 
        Approximate the actual leaf sizes and the projection factor throughout the day 
        '''
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
            terminal_step = (observed_area.shape[0]-2)*self.steps_per_day - 1

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

        # Optionally set a larger time step than the one given in plant data
        assert self.steps_per_day % self.stride == 0 & self.steps_per_night % self.stride == 0, f"stride must be a divisor of steps_per_day={self.steps_per_day} and of steps_per_night={self.steps_per_night}."
        PF = [pf[::self.stride] for pf in PF]
        PWL = [pwl.rescale_x(1 / self.stride) for pwl in PWL]
        self.steps_per_day /= self.stride
        self.steps_per_night /= self.stride
        self.interval *= self.stride
        terminal_step = int(terminal_step / self.stride)

        return PWL, PF, terminal_step
    
    def load_area_data(self):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/plant_area_data.csv"
        df = pd.read_csv(data_path).sort_values(by='timestamp')

        # Filter out any plants that have sensor reading errors (area randomly goes to 0 at some timesteps)
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
        steps_per_night = int((night_duration.mode()[0] / time_increment)-1)

        # The second when the first day starts 
        first_second = pd.to_datetime(df['timestamp'].iloc[0]).time().hour * 3600 + pd.to_datetime(df['timestamp'].iloc[0]).time().minute * 60

        # Observed areas of plants (in unit of pixels)
        plant_area_data = np.array(df.drop(columns=['timestamp']))
        assert plant_area_data.shape[1] >= self.num_plants, f"Please request fewer than {plant_area_data.shape[1]} plants."

        return plant_area_data[:,:self.num_plants], steps_per_day, steps_per_night, time_increment.total_seconds(), first_second
    
    def sine_time(self, t):
        # Return sine & cosine times, normalized to between 0 and 1
        return [(sin(2*pi*t/86400)+1)/2, (cos(2*pi*t/86400)+1)/2]
    
    def normalize(self, x):   # normalize observation to between 0 and 1
        u = 30000   # max historic area of one plant (in pixels)
        l = 0
        if isinstance(x, list):
            return [(val - l) / (u - l) for val in x]
        return (x - l) / (u - l)