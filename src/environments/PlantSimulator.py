import os
from math import sin, cos, pi
import numpy as np
import pandas as pd
from RlGlue.environment import BaseEnvironment
from utils.functions import PiecewiseLinear
from utils.metrics import UnbiasedExponentialMovingAverage as uema
import jax.numpy as jnp      

class PlantSimulator(BaseEnvironment):  
    ''' 
    Simulate a tray of plants under the same lighting agent.
    State = (sin time-of-day, cos time-of-day, sin countdown, cos countdown, average area, history of change in average area)
    Action = [moonlight, low, med, high] (med is optimal at noon, high is too bright)
    Reward = change in average area over 1 step
    '''
    def __init__(self, num_plants=48, outliers=2, lag=1, stride=1, last_day=14, **kwargs):
        self.state_dim = (6,)   
        self.current_state = np.empty(6)
        self.action_dim = 4
        self.actions = [0, 1, 2, 3] 

        self.num_plants = num_plants
        self.outliers = outliers        # number of "outliers" to remove from the top/bottom observations
        
        self.stride = stride            # env time step = stride * time step in plant data
        self.lag = lag                  # lag for change in area used in reward function; default is 1 time step

        self.observed_areas = []        # stores a list of lists of daytime observed areas in pixels. i.e. self.observed_areas[-1] contains the latest areas of individual plants
        
        self.history = uema(alpha=0.01) # history of change in average observed area over 1 step (in units of pixels)
        
        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.num_steps = 0
        self.last_day = last_day        # horizon in units of days
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today
        
        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_areas, self.projection_factors, self.terminal_step = self.analyze_area_data()

        self.gamma = 1.0
        
    def start(self):
        self.frozen_time_today = 0
        self.time = 0
        self.num_steps = 0

        self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]   # Make a copy because actual_areas will be modified at each step
        
        self.observed_areas = [[self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] for i in range(self.num_plants)]]
        
        self.history = uema(alpha=0.01)
        
        self.current_state = np.hstack([self.time_of_day(),
                                        self.countdown(),
                                        self.normalize(self.iqm(self.observed_areas[-1]), l=0, u=15000),
                                        self.normalize(0, l=-5, u=30)])   # let the history be zero at t=0

        return self.current_state

    def step(self, action):
        # Check if the agent selected optimal action given the time of day. This is env specific and should be overwritten by subclasses. 
        self.last_action_optimal = self.is_optimal(action)
        
        # Modify the interpolated actual_areas according to the action
        for pwl in self.actual_areas: 
            pwl.insert_plateau(self.time, self.time + self.frozen_time(action))
        self.frozen_time_today += self.frozen_time(action)
        
        # Keep track of time
        self.time += 1
        self.num_steps += 1

        # Overnight behaviors
        if self.num_steps % self.steps_per_day == 0:  # if transitioning into the next day
            # freeze growth at night by the same amount as daytime today
            for pwl in self.actual_areas:
                pwl.insert_plateau(self.time, self.time + self.frozen_time_today)
                
            # compute history throughout night time
            self.overnight_trace()
                           
            self.time += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        # Compute observed areas by projecting actual areas
        self.observed_areas.append([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] 
                                    for i in range(self.num_plants)])
        
        # Compute reward
        self.reward = self.reward_function()
        
        # history = trace over "change in iqm(observed areas)" over 1 time step
        if self.num_steps % self.steps_per_day != 0:    # note: the overnight case is as above
            self.history.update(self.iqm(self.observed_areas[-1]) - self.iqm(self.observed_areas[-2]))
        
        self.current_state = np.hstack([self.time_of_day(),
                                        self.countdown(),
                                        self.normalize(self.iqm(self.observed_areas[-1]), l=0, u=15000),
                                        self.normalize(self.history.compute(), l=-5, u=30)])

        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
    
    def overnight_trace(self): 
        last_night_obs = self.observed_areas[-1]
        overnight_obs = []
        for i in range(self.num_plants):
            # Interpolate for projection factors overnight
            last_night_pf = self.projection_factors[i][self.num_steps - 1] 
            morning_pf = self.projection_factors[i][self.num_steps] 
            delta_pf = (morning_pf - last_night_pf) / self.steps_per_night
            # Compute overnight observations
            overnight_ob = [last_night_obs[i]]
            for j in range(int(self.steps_per_night)):
                pf = last_night_pf + delta_pf * (j + 1)
                overnight_ob.append(self.actual_areas[i](self.time + j) * pf)
            overnight_obs.append(overnight_ob)
            
        overnight_obs = np.array(overnight_obs).T   
        
        for j in range(int(self.steps_per_night)):   
            self.history.update(self.iqm(overnight_obs[j + 1]) - self.iqm(overnight_obs[j]))           
                
    def iqm(self, values):
        values = np.sort(values)
        return np.mean(values[self.outliers:self.num_plants-self.outliers])
        
    def reward_function(self):
        if self.num_steps >= self.lag: 
            new = self.normalize(self.iqm(self.observed_areas[-1]), l=0, u=15000)
            old = self.normalize(self.iqm(self.observed_areas[-1-self.lag]), l=0, u=15000)
            return new - old
        else: 
            return 0

    def get_info(self):
        return {"gamma": self.gamma, 'action_is_optimal': self.last_action_optimal}
        
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
        assert plant_area_data.shape[1] >= self.num_plants, f"Please request fewer than {plant_area_data.shape[1]} plants."
        
        # Check if we have enough data for the requested last_day 
        assert plant_area_data.shape[0] / steps_per_day - 2 >= self.last_day, f'The requested last_day exceeds available plant data, which has {plant_area_data.shape[0] / steps_per_day - 2} days.'

        return plant_area_data[:,:self.num_plants], steps_per_day, steps_per_night, time_increment.total_seconds(), first_second
    
    def time_of_day(self):  # Return sine & cosine times, normalized to between 0 and 1
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds
        return [(sin(2*pi*clock/86400)+1)/2, (cos(2*pi*clock/86400)+1)/2]
    
    def countdown(self):
        total_steps = self.last_day * self.steps_per_day
        return [(sin(2*pi*self.num_steps/total_steps)+1)/2, (cos(2*pi*self.num_steps/total_steps)+1)/2]
    
    def normalize(self, x, l, u):   # normalize areas to between 0 and 1 
        if isinstance(x, list):
            return [(val - l) / (u - l) for val in x]
        return (x - l) / (u - l) 
    
    def frozen_time(self, action):       
        # Amount of frozen time (in unit of time step), given action
        twilight = {0: 1.0, 1: 0.0, 2: 0.5, 3: 1.0}  # rank from good to bad: low, med, off/high
        noon = {0: 1.0, 1: 0.5, 2: 0.0, 3: 1.0}      # rank from good to bad: med, low, off/high
        
        clock = (self.num_steps % self.steps_per_day)*self.interval    # seconds since beginning of day
        total_seconds = self.steps_per_day*self.interval               # total seconds during day time  
        
        assert self.steps_per_day % 4 == 0, 'steps_per_day needs to be divisible by 4 in the current implementation of "frozen_time".'
        if clock < 1/4*total_seconds or clock >= 3/4*total_seconds:   
            return twilight[action]
        else: 
            return noon[action]
    
    def is_optimal(self, action):
        clock = (self.num_steps % self.steps_per_day)*self.interval   
        total_seconds = self.steps_per_day*self.interval             
        if clock < 1/4*total_seconds or clock >= 3/4*total_seconds:   
            return action == 1   
        else:
            return action == 2
        
class PlantSimulator_Only1Time(PlantSimulator):  
    ''' 
    State = (linear time-of-day, average area, history of change in average area)
    Action = [moonlight, low, med, high] (med is optimal at noon, high is too bright)
    Reward = change in average area over 1 step
    '''
    def __init__(self, num_plants=48, outliers=2, lag=1, stride=1, last_day=14, **kwargs):
        super().__init__(num_plants, outliers, lag, stride, last_day)
        self.state_dim = (3,)   
        self.current_state = np.empty(3)
    
    def start(self):
        super().start()
        self.current_state = np.hstack([self.linear_time_of_day(),
                                        self.normalize(self.iqm(self.observed_areas[-1]), l=0, u=15000),
                                        self.normalize(0, l=-5, u=30)])   
        return self.current_state
    
    def step(self, action):
        super().step(action)
        self.current_state = np.hstack([self.linear_time_of_day(),
                                        self.normalize(self.iqm(self.observed_areas[-1]), l=0, u=15000),
                                        self.normalize(self.history.compute(), l=-5, u=30)])
        
        if self.num_steps == self.terminal_step:
            return self.reward, self.current_state, True, self.get_info()
        else:    
            return self.reward, self.current_state, False, self.get_info()
        
    def linear_time_of_day(self):
        step_today = self.num_steps % self.steps_per_day  
        return step_today / self.steps_per_day
    

class PlantSimulator_Only1Time_EMAReward(PlantSimulator_Only1Time):
    '''
    State = (linear time-of-day, average area, history of change in average area)
    Action = [moonlight, low, med, high] (med is optimal at noon, high is too bright)
    Reward = change in fast exponential moving average of area and slow exponential moving average of area
    '''
    def __init__(self, num_plants=48, outliers=2, lag=1, stride=1, last_day=14, **kwargs):
        super().__init__(num_plants, outliers, lag, stride, last_day, **kwargs)
        self.area_history_fast = uema(alpha=0.6)
        self.area_history_slow = uema(alpha=0.06)

    def reward_function(self):
        r = self.area_history_fast.compute() - self.area_history_slow.compute()
        # normalize
        r /= 150
        return r.item()

    def step(self, action):
        output = super().step(action)
        self.area_history_fast.update(self.iqm(self.observed_areas[-1]))
        self.area_history_slow.update(self.iqm(self.observed_areas[-1]))
        return output

class PlantSimulatorLowHigh(PlantSimulator):
    '''
    Simulate a tray of plants under the same lighting agent.
    State = (sin time-of-day, cos time-of-day, sin countdown, cos countdown, average area, history of change in average area)
    Action = [low, high]
    "low" is optimal in twilight hours, "high" is optimal otherwise.
    Reward = change in average area over 1 step
    '''
    def __init__(self, num_plants=48, outliers=2, lag=1, stride=1, last_day=14, **kwargs):
        super().__init__(num_plants, outliers, lag, stride, last_day)
        self.action_dim = 2 
        self.actions = [0, 1]
    
    def frozen_time(self, action):       
        # Amount of frozen time (in unit of time step), given action
        twilight = {0: 0.0, 1: 0.5}
        noon = {0: 0.5, 1: 0.0}
        
        clock = (self.num_steps % self.steps_per_day)*self.interval    # seconds since beginning of day
        total_seconds = self.steps_per_day*self.interval               # total seconds during day time  
        if clock < 0.25*total_seconds or clock > 0.75*total_seconds:   # if during the first or last 25% of daytime
            return twilight[action]
        else: 
            return noon[action]
    
    def is_optimal(self, action):
        clock = (self.num_steps % self.steps_per_day)*self.interval    # seconds since beginning of day
        total_seconds = self.steps_per_day*self.interval               # total seconds during day time  
        if clock < 0.25*total_seconds or clock > 0.75*total_seconds:   # twilight
            return action == 0   # twilight
        else:
            return action == 1   # near noon