import numpy as np
from RlGlue.environment import BaseEnvironment

import logging
logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)

class CleanPlantSimulator(BaseEnvironment):
    '''
    Simulate the evolution of plant area based on lighting conditions.
    Assumptions: (i) Plant motion throughout each day is the same gaussian curve.
                 (ii) Growth only occurs at night. Overnight growth is 30% if lighting is standard.
                 (iii) Changing lighting only affects overnight growth.
    State = [time of day, area, daily light integral (dli)] (all normalized to [0,1])
    Action = [low, standard]
    Reward = overnight growth assigned to last time stamp of each day 
    Time Step = 10 min
    Episode duration = 14 days
    '''
    def __init__(self, **kwargs):
        self.state_dim = (3,)
        self.current_state = np.empty(3)
        self.action_dim = 2
        self.actions = [0, 1]

        self.steps_per_day = 72
        self.total_steps = 72*14
        
        self.steps = 0
        self.dli = 0
        self.current_morning_size = 1

        self.gamma = 1.0

    def get_observation(self):
        area = self.current_morning_size + self.daily_area_curve(self.steps % self.steps_per_day)
        tod = self.steps % self.steps_per_day
        daily_light_integral = self.dli

        observation = np.hstack([self.normalize(tod, 0, self.steps_per_day),
                                 self.normalize(area, 1, 14),
                                 self.normalize(daily_light_integral, 0, self.steps_per_day)])
        return observation
    
    def start(self):
        self.steps = 0
        self.current_morning_size = 1
        self.dli = 0

        self.current_state = self.get_observation()

        return self.current_state, self.get_info()

    def step(self, action):
        self.steps += 1
        self.dli += action
        
        if self.steps % self.steps_per_day == 0:    # if transitioning overnight
            overnight_growth = self.current_morning_size * 0.2 * self.normalize(self.dli, 0, self.steps_per_day)
            self.current_morning_size += overnight_growth
            self.reward = overnight_growth
            self.dli = 0
        else: 
            self.reward = 0
 
        self.current_state = self.get_observation()

        if self.steps == self.total_steps:
            return self.reward, self.current_state, True, self.get_info()
        else:
            return self.reward, self.current_state, False, self.get_info()

    def get_info(self):
        return {"gamma": self.gamma}

    def normalize(self, x, l, u):
        return (x - l) / (u - l)
    
    def daily_area_curve(self, x):
        """
        Model the daily area curve by a Gaussian curve. 
        Input: x is the number of steps so far today.
        """
        stdev = self.steps_per_day
        mu = self.steps_per_day / 2
        gaussian = self.current_morning_size * np.exp(-0.5 * ((x - mu) / stdev) ** 2)
        shift = self.current_morning_size * np.exp(-0.5 * ((0 - mu) / stdev) ** 2)
        return gaussian - shift
    

class CleanPlantSimulator_Daily(CleanPlantSimulator):
    '''
    Scale down the observed area and the overnight reward so that they are consistent throughout 14 days
    State = [time of day, daily light integral (dli), daily area] (all normalized to [0,1])
    Reward = percentage overnight growth assigned to last time stamp of each day 
    '''
    def get_observation(self):
        area = self.current_morning_size + self.daily_area_curve(self.steps % self.steps_per_day)

        tod = self.steps % self.steps_per_day
        daily_area = area / self.current_morning_size   # daily area is area scaled by morning area
        daily_light_integral = self.dli

        observation = np.hstack([self.normalize(tod, 0, self.steps_per_day),
                                 self.normalize(daily_area, 1, 1.12),
                                 self.normalize(daily_light_integral, 0, self.steps_per_day)])
        return observation

    def step(self, action):
        self.steps += 1
        self.dli += action
        
        if self.steps % self.steps_per_day == 0:    # if transitioning overnight
            percent_overnight_growth = 0.2 * self.normalize(self.dli, 0, self.steps_per_day)
            self.current_morning_size += self.current_morning_size * percent_overnight_growth
            self.reward = percent_overnight_growth
            self.dli = 0
        else: 
            self.reward = 0
 
        self.current_state = self.get_observation()

        if self.steps == self.total_steps:
            return self.reward, self.current_state, True, self.get_info()
        else:
            return self.reward, self.current_state, False, self.get_info()