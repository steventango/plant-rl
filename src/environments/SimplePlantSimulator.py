import os
import numpy as np
import pandas as pd
from RlGlue.environment import BaseEnvironment
from utils.functions import PiecewiseLinear
from math import floor
import datetime
from utils.metrics import UnbiasedExponentialMovingAverage as uema


import logging
logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)

class SimplePlantSimulator(BaseEnvironment):
    '''
    Simulate a tray of plants under the same lighting agent.
    State = [linear time-of-day,
             average area,
             percentage change in average area over 1 step (for tracking plant motion)
            ] 
    Action = [low, standard]
    Reward = raw change in average area over 1 step (includes large overnight growth)
    '''
    def __init__(self, reward_label, **kwargs):
        self.state_dim = (3,)
        self.current_state = np.empty(3)
        self.action_dim = 2
        self.actions = [0, 1]

        self.observed_areas = []        # stores a list of lists of daytime observed areas in pixels. i.e. self.observed_areas[-1] contains the latest areas of individual plants

        self.all_steps = 0              # step counter that counts both day and night, even though agent sleeps at night
        self.day_steps = 0
        self.frozen_time_today = 0      # how long the plant has be frozen during daytime today
        self.num_plants = 0             # num_plants will be set as the total number of plants in the data

        self.data, self.steps_per_day, self.steps_per_night, self.interval, self.first_second = self.load_area_data()
        self.original_actual_areas, self.projection_factors = self.analyze_area_data()

        self.gamma = 1.0

        self.reward_label = reward_label

    def get_observation(self, new_ob=True):
        # Append new observation if new_ob
        if new_ob:
            self.observed_areas.append(np.array([self.actual_areas[i](self.all_steps)*self.projection_factors[i][self.day_steps] for i in range(self.num_plants)]))

        # Compute the average of the current observed areas
        new_area = np.mean(self.observed_areas[-1])

        # Compute plant motion, defined as percent change in average area per step
        if self.day_steps == 0:
            plant_motion = 0
        elif self.day_steps % self.steps_per_day == 0:   # calculate % growth per overnight step, assuming constant % growth
            old_area = np.mean(self.observed_areas[-2])
            total_growth = new_area / old_area - 1
            plant_motion = (total_growth + 1)**(1 / self.steps_per_night) - 1
        else:
            old_area = np.mean(self.observed_areas[-2])
            plant_motion = new_area / old_area - 1

        observation = np.hstack([self.time_of_day(),
                                 self.normalize(new_area, l=0, u=600),
                                 self.normalize(plant_motion, l=-0.05, u=0.05)])

        return observation

    def start(self):
        self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]   # Make a copy because actual_areas will be modified at each step
        self.frozen_time_today = 0
        self.all_steps = 0
        self.day_steps = 0
        self.observed_areas = []

        self.current_state = self.get_observation()

        return self.current_state, self.get_info()

    def step(self, action):
        # Modify the interpolated actual_areas according to the action
        for pwl in self.actual_areas:
            pwl.insert_plateau(self.all_steps, self.all_steps + self.frozen_time(action))
        self.frozen_time_today += self.frozen_time(action)

        # Keep track of time
        self.all_steps += 1   # must occur after the above action effect
        self.day_steps += 1

        # Overnight behaviors
        if self.day_steps % self.steps_per_day == 0:  # if transitioning into the next day
            # freeze growth at night by the same amount as daytime today
            for pwl in self.actual_areas:
                pwl.insert_plateau(self.all_steps, self.all_steps + self.frozen_time_today)
            self.all_steps += self.steps_per_night - 1   # fastforward time
            self.frozen_time_today = 0

        self.current_state = self.get_observation()

        self.reward = self.reward_function()

        if self.day_steps == self.total_data_steps:
            return self.reward, self.current_state, True, self.get_info()
        else:
            return self.reward, self.current_state, False, self.get_info()

    def reward_function(self):
        new_area = np.mean(self.observed_areas[-1])
        old_area = np.mean(self.observed_areas[-2])
        if self.reward_label == 'r1':
            return self.normalize(new_area - old_area, 0, 50)
        elif self.reward_label == 'r1_percent':
            return self.normalize(new_area / old_area - 1, 0, 0.25)
        elif self.reward_label == 'daily_percent':
            if self.day_steps % self.steps_per_day == 0 and floor(self.day_steps / self.steps_per_day) >= 1:
                new_area = np.mean(self.observed_areas[-1])
                old_area = np.mean(self.observed_areas[-1-self.steps_per_day])
                return self.normalize(new_area / old_area - 1, 0, 0.35)
            else:
                return 0.0
        else:
            raise ValueError(f"{self.reward_label} is an invalid reward_label for the SimplePlantSimulator class.")

    def get_info(self):
        return {"gamma": self.gamma}

    def time_of_day(self):
        step_today = self.day_steps % self.steps_per_day
        return step_today / self.steps_per_day

    def normalize(self, x, l, u):
        return (x - l) / (u - l)

    def frozen_time(self, action):
        # Amount of frozen time (in unit of time step), given action
        all_time = {0: 1.0, 1: 0.0}
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
                max_time.append((i-1)*(self.steps_per_day+self.steps_per_night) + max_indices[i])  # Let time=0 be at the first time stamp of day 2
                max_area.append(ep_data[i*self.steps_per_day + max_indices[i]])
            pwl = PiecewiseLinear(max_time, max_area)

            # Number of remaining daytime time stamps (since we truncate the first and last day)
            self.total_data_steps = (observed_area.shape[0]-2)*self.steps_per_day

            # Compute the actual area at all daytime time stamps
            actual_area_daytime = []
            for i in range(observed_area.shape[0]-2):
                x_values = np.arange(i*(self.steps_per_day+self.steps_per_night), i*(self.steps_per_day+self.steps_per_night)+self.steps_per_day)
                y_values = [pwl(x) for x in x_values]
                actual_area_daytime.append(y_values)
            last_x = (observed_area.shape[0]-2)*(self.steps_per_day+self.steps_per_night)
            actual_area_daytime.append(pwl(last_x))
            actual_area_daytime = np.hstack(actual_area_daytime)

            # Compute projection factor
            truncated_data = ep_data[self.steps_per_day:-self.steps_per_day + 1]
            projection_factor = truncated_data / actual_area_daytime
            PWL.append(pwl)
            PF.append(projection_factor)

        return PWL, PF

    def load_area_data(self, time_increment = 10):
        # Load historic plant area data
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/plant_data/reprocessed.csv"
        full_df = pd.read_csv(data_path)

        # Number of plants in the dataset
        self.num_plants = full_df['plant_id'].max()

        # Remove night time filler data
        full_df['time'] = pd.to_datetime(full_df['time'])
        full_df['time_only'] = full_df['time'].dt.time
        start_time = datetime.time(9, 25)
        end_time = datetime.time(20, 45)
        df = full_df[(full_df['time_only'] >= start_time) & (full_df['time_only'] <= end_time)]

        # Number of time steps per day
        timestamps_per_day = df['time'].dt.date.value_counts() / self.num_plants
        if timestamps_per_day.nunique() != 1:
            raise ValueError(f"Inconsistent timestamps per day: {timestamps_per_day.to_dict()}")
        steps_per_day = int(timestamps_per_day.iloc[0])

        # Number of time steps per night
        first_date = df['time'].dt.date.min()
        second_date = df[df['time'].dt.date > first_date]['time'].dt.date.min()
        night_duration = df[df['time'].dt.date == second_date]['time'].min() - df[df['time'].dt.date == first_date]['time'].max()
        steps_per_night = int((int(night_duration.total_seconds()/60) / time_increment) - 1)

        # The second when the first day starts
        first_second = pd.to_datetime(df['time'].iloc[0]).time().hour * 3600 + pd.to_datetime(df['time'].iloc[0]).time().minute * 60

        # Observed areas of plants (in unit of pixels)
        plant_area_data = df['clean_area'].to_numpy()
        plant_area_data = np.reshape(plant_area_data, (-1, self.num_plants))

        return plant_area_data, steps_per_day, steps_per_night, time_increment*60, first_second

class TOD_action(SimplePlantSimulator):
    '''
    State = (TOD, action history)
    '''
    def __init__(self, reward_label, **kwargs):
        super().__init__(reward_label, **kwargs)
        self.state_dim = (2,)
        self.current_state = np.empty(2)
        self.action_history = uema(alpha=0.1) # trace of actions

    def start(self):
        self.action_history.reset()
        return super().start()

    def get_observation(self, new_ob=True):
        super().get_observation(new_ob)
        return np.array([self.time_of_day(), self.action_history.compute().item()])

    def step(self, action):
        self.action_history.update(action)
        return super().step(action)

class Daily_Sim(SimplePlantSimulator):
    '''
    Start a new episode every day.
    '''
    def start(self):
        if self.day_steps == 0 or self.day_steps == self.total_data_steps:
            self.actual_areas = [pwl.copy() for pwl in self.original_actual_areas]
            self.frozen_time_today = 0
            self.all_steps = 0
            self.day_steps = 0
            self.observed_areas = []

        if self.day_steps % self.steps_per_day == 0 and self.day_steps > 0:
            self.current_state = self.get_observation(new_ob=False)   # reuse last step of previous episode (morning) as the first step of current episode
        else:
            self.current_state = self.get_observation(new_ob=True)

        return self.current_state, self.get_info()

    def step(self, action):
        super().step(action)
        if self.day_steps % self.steps_per_day == 0:
            return self.reward, self.current_state, True, self.get_info()
        else:
            return self.reward, self.current_state, False, self.get_info()

    def reward_function(self):
        new_area = np.mean(self.observed_areas[-1])
        old_area = np.mean(self.observed_areas[-2])
        if self.reward_label == 'r1':
            return self.normalize(new_area - old_area, 0, 50)
        elif self.reward_label == 'r1_percent':
            return self.normalize((new_area - old_area) / old_area, 0, 0.25)
        elif self.reward_label == 'r1_norm':
            first_stamp_of_day = floor(self.day_steps / self.steps_per_day) * self.steps_per_day
            first_area_of_day = np.mean(self.observed_areas[first_stamp_of_day])
            return self.normalize((new_area - old_area) / first_area_of_day, 0, 0.2)
        elif self.reward_label == 'r2_1day':
            n = self.steps_per_day
            if self.day_steps % n == 0 and self.day_steps >= 2*n:
                new = np.mean(self.observed_areas[-n:])
                old = np.mean(self.observed_areas[-2*n:-n])
                return self.normalize(new / old - 1, 0, 0.25)
            else:
                return 0
        elif self.reward_label == 'r2_3hr':
            n = 17   # neater than 18 because steps_per_day is 68
            if self.day_steps % n == 0 and self.day_steps >= 2*n:
                new = np.mean(self.observed_areas[-n:])
                old = np.mean(self.observed_areas[-2*n:-n])
                return self.normalize(new / old - 1, 0, 0.25)
            else:
                return 0
        else:
            raise ValueError(f"{self.reward_label} is an invalid reward_label for the Daily_Sim class.")

    def get_info(self):
        return {"gamma": self.gamma, "day_steps": self.day_steps, "steps_per_day": self.steps_per_day}

class Daily_Bandit(Daily_Sim):
    '''
    State = trivial
    Overnight reward = 0
    '''
    def __init__(self, reward_label, **kwargs):
        super().__init__(reward_label, **kwargs)
        self.state_dim = (1,)
        self.current_state = np.empty(1)
        self.gamma = 0.0

    def get_observation(self, new_ob=True):
        super().get_observation(new_ob)
        return np.array([1])

    def reward_function(self):
        if self.day_steps % self.steps_per_day == 0:
            return 0
        else:
            return super().reward_function()

class Daily_ContextBandit(Daily_Sim):
    '''
    State = TOD
    '''
    def __init__(self, reward_label, **kwargs):
        super().__init__(reward_label, **kwargs)
        self.state_dim = (1,)
        self.current_state = np.empty(1)
        self.gamma = 0.0

    def get_observation(self, new_ob=True):
        super().get_observation(new_ob)
        return np.array([floor(self.day_steps % self.steps_per_day / 6)])

    def step(self, action):
        super().step(action)
        if self.day_steps % self.steps_per_day == 0:
            self.current_state = np.array([12])
            return self.reward, self.current_state, True, self.get_info()
        else:
            return self.reward, self.current_state, False, self.get_info()

class Daily_ESARSA_TOD(Daily_ContextBandit):
    def __init__(self, reward_label, **kwargs):
        super().__init__(reward_label, **kwargs)
        self.gamma = 1.0
