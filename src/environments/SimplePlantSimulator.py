import os
from math import sin, cos, pi
import numpy as np
import pandas as pd
from RlGlue.environment import BaseEnvironment
from utils.functions import PiecewiseLinear
from utils.metrics import UnbiasedExponentialMovingAverage as uema
from utils.metrics import iqm

class SimplePlantSimulator(BaseEnvironment):
    '''
    Simulate a tray of plants under the same lighting agent.
    State = (linear time-of-day, history of symmetric % change in average area)
    Action = [moonlight, low, med, high] (med is optimal at noon, high is too bright)
    Reward = history of symmetric % change in average area
    '''
    def __init__(self, num_plants=48, q=0.05, stride=1, l=0.0005, u=0.0025, last_day=14, **kwargs):
        self.state_dim = (2,)
        self.current_state = np.empty(2)
        self.action_dim = 4
        self.actions = [0, 1, 2, 3]

        self.num_plants = num_plants
        self.q = q                      # the bottom q and the top 1-q quantiles are excluded from iqm
        self.stride = stride            # env time step = stride * time step in plant data
        self.l = l                      # lower bound for normalizing growth rate to [0,1]
        self.u = u                      # upper bound ...

        self.observed_areas = []        # stores a list of lists of daytime observed areas in pixels. i.e. self.observed_areas[-1] contains the latest areas of individual plants

        self.history = uema(alpha=0.01) # history of % change in average observed area over 1 step (in units of %)

        self.time = 0                   # step counter that counts both day and night, even though agent is sleeping at night
        self.num_steps = 0
        self.last_day = last_day        # horizon in units of days
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today

        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_areas, self.projection_factors, self.terminal_step = self.analyze_area_data()

        self.gamma = 0.99
    
    def get_observation(self):
        # Compute observed areas by projecting actual areas
        self.observed_areas.append(np.array([self.actual_areas[i](self.time)*self.projection_factors[i][self.num_steps] for i in range(self.num_plants)]))        
        
        # Compute history of % change in average area
        if self.num_steps % self.steps_per_day != 0:  # DO NOT trace overnight here!
            old_area = iqm(self.observed_areas[-2], self.q)
            new_area = iqm(self.observed_areas[-1], self.q)
            self.history.update(self.percent_change(old_area, new_area))
        
        observation = np.hstack([self.time_of_day(), self.normalize(self.history.compute())])
            
        return observation

    def start(self):
        self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]   # Make a copy because actual_areas will be modified at each step

        self.frozen_time_today = 0
        self.time = 0
        self.num_steps = 0
        self.observed_areas = []
        self.history.reset()

        self.current_state = self.get_observation()

        return self.current_state

    def step(self, action):
        # Check if the agent selected optimal action given the time of day. This is env specific and should be overwritten by subclasses.
        self.last_action_optimal = self.is_optimal(action)

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

            # compute history throughout night time
            self.overnight_trace()

            self.time += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        self.current_state = self.get_observation()
        
        self.reward = self.reward_function(action)

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
            delta_pf = (morning_pf - last_night_pf) / (self.steps_per_night + 1)
            
            # Compute overnight observations
            overnight_ob = [last_night_obs[i]]
            for j in range(int(self.steps_per_night) + 1):
                pf = last_night_pf + delta_pf * (j + 1)
                overnight_ob.append(self.actual_areas[i](self.time + j) * pf)
            overnight_obs.append(overnight_ob)

        overnight_obs = np.array(overnight_obs).T

        for j in range(int(self.steps_per_night) + 1):
            old_area = iqm(overnight_obs[j], self.q)
            new_area = iqm(overnight_obs[j + 1], self.q)
            self.history.update(self.percent_change(old_area, new_area))

    def reward_function(self, action):
        return self.current_state[-1]

    def get_info(self):
        return {"gamma": self.gamma, 'action_is_optimal': self.last_action_optimal}

    def time_of_day(self):
        step_today = self.num_steps % self.steps_per_day
        return step_today / self.steps_per_day

    def normalize(self, x):  
        return (x - self.l) / (self.u - self.l)
    
    def percent_change(self, old, new):   # symmetric percentage change
        return 2 * (new - old) / (new + old)

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
    

class TrivialRewEnv(SimplePlantSimulator):
    '''
    Uses trivial +1 or -1 reward for following the twilight policy
    '''
    def __init__(self, num_plants=48, q=0.05, stride=1, l=0.0005, u=0.0025, last_day=14, **kwargs):
        super().__init__(num_plants, q, stride, l, u, last_day, **kwargs)     

    def reward_function(self, action):
        clock = (self.num_steps % self.steps_per_day)*self.interval    # seconds since beginning of day
        total_seconds = self.steps_per_day*self.interval               # total seconds during day time
        twilight = clock < 1/4*total_seconds or clock >= 3/4*total_seconds
        noon = not twilight
        if (twilight and action==1) or (noon and action == 2):
            return 1.0
        else:
            return -1.0

class SineTimeEnv(SimplePlantSimulator):
    '''
    Uses sine/cos time encoding
    '''
    def __init__(self, num_plants=48, q=0.05, stride=1, l=0.0005, u=0.0025, last_day=14, **kwargs):
        super().__init__(num_plants, q, stride, l, u, last_day, **kwargs)     
        self.state_dim = (3,)
        self.current_state = np.empty(3)

    def time_of_day(self):
        clock = (self.num_steps % self.steps_per_day)*self.interval + self.first_second   # time of day in seconds
        return [(sin(2*pi*clock/86400)+1)/2, (cos(2*pi*clock/86400)+1)/2]
    

class TrivialRewSineTimeEnv(SineTimeEnv, TrivialRewEnv):
    '''
    Uses sine/cos time encoding and trivial +1 or -1 reward for following the twilight policy
    '''
    def __init__(self, num_plants=48, q=0.05, stride=1, l=0.0005, u=0.0025, last_day=14, **kwargs):
        super().__init__(num_plants, q, stride, l, u, last_day, **kwargs)
    