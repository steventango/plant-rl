import logging

import numpy as np
from RlGlue.environment import BaseEnvironment

logger = logging.getLogger("rlglue")
logger.setLevel(logging.DEBUG)


class CleanPlantSimulator(BaseEnvironment):
    """
    Simulate the evolution of plant area based on lighting conditions.
    Assumptions: (i) Plant motion throughout each day is modeled by a gaussian curve.
                 (ii) Growth only occurs at night. Overnight growth is 20% if lighting is standard.
                 (iii) Poor lighting reduces overnight growth and affects the shape of the gaussian curve.
    State = [time of day,
             plant area,
             plant openness,
             daily light integral (dli)] (all normalized to [0,1])
    Action = 1D continuous intensity. 0 is off. 1 is standard.
    Reward = percentage or raw overnight growth assigned to last time stamp of each day
    """

    def __init__(self, **kwargs):
        self.state_dim = (4,)
        self.current_state = np.empty(4)

        self.time_step = 5  # minutes
        self.duration = 14  # days
        self.steps_per_day = int(12 * 60 / self.time_step)
        self.total_steps = int(self.steps_per_day * self.duration)

        self.steps = 0
        self.dli = 0
        self.current_morning_size = 1

        self.gamma = 1.0

        self.daily_reward = kwargs.get("daily_reward", True)

    def get_observation(self):
        area = self.current_morning_size + self.daily_area_curve(
            self.steps % self.steps_per_day
        )

        tod = self.steps % self.steps_per_day
        daily_light_integral = self.dli
        openness = area / self.current_morning_size

        observation = np.hstack(
            [
                self.normalize(tod, 0, self.steps_per_day),
                self.normalize(area, 0.5, 14),
                self.normalize(openness, 0.97, 1.15),
                self.normalize(daily_light_integral, 0, self.steps_per_day),
            ]
        )

        return observation

    def get_reward(self, percent_overnight_growth):
        if self.daily_reward:
            return self.normalize(percent_overnight_growth, 0, 0.2)
        else:
            return self.normalize(
                self.current_morning_size * percent_overnight_growth, 0, 2
            )

    def get_light_amount(self, action):
        return action

    def start(self):
        self.steps = 0
        self.current_morning_size = 1
        self.dli = 0

        self.current_state = self.get_observation()

        return self.current_state, self.get_info()

    def step(self, action):
        self.steps += 1
        self.dli += self.get_light_amount(action)

        if self.steps % self.steps_per_day == 0:  # if transitioning overnight
            if self.dli <= self.steps_per_day:
                percent_overnight_growth = 0.2 * self.normalize(
                    self.dli, 0, self.steps_per_day
                )
            else:
                percent_overnight_growth = 0.2 * self.normalize(
                    2 * self.steps_per_day - self.dli, 0, self.steps_per_day
                )
            self.reward = self.get_reward(percent_overnight_growth)
            self.current_morning_size += (
                self.current_morning_size * percent_overnight_growth
            )
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

    def normalize(self, x, lower, upper):
        return (x - lower) / (upper - lower)

    def daily_area_curve(self, x):
        """
        Model the daily area curve by a Gaussian curve.
        Input: x is the number of steps so far today.
        """
        slowing = 1 + (x - self.dli) / self.steps_per_day
        x = x / slowing

        stdev = self.steps_per_day
        mu = self.steps_per_day / 2
        gaussian = self.current_morning_size * np.exp(-0.5 * ((x - mu) / stdev) ** 2)
        shift = self.current_morning_size * np.exp(-0.5 * ((0 - mu) / stdev) ** 2)
        return gaussian - shift
