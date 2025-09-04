import logging

import numpy as np
import datetime
from collections import defaultdict
from utils.RlGlue.environment import BaseAsyncEnvironment

logger = logging.getLogger("rlglue")
logger.setLevel(logging.DEBUG)


class CleanPlantSimulator(BaseAsyncEnvironment):
    """
    Simulate the dynamics of plant area under changing lighting condition.
    Assumptions: (i) Plant motion throughout each day is modeled by a gaussian curve.
                 (ii) Growth only occurs at night. Overnight growth is 20% if lighting is optimal.
                 (iii) Suboptimal lighting reduces overnight growth and affects the shape of the gaussian curve.
    State = [local time, mean plant area]
    Action = intensity in the range [0 ppfd, 150 ppfd]. 100 ppfd is optimal. Spectral composition is standard.
    Reward = percentage or raw overnight growth assigned to last time stamp of each day
    """

    def __init__(self, **kwargs):
        self.state_dim = (2,)
        self.current_state = np.empty(2)

        self.percent_reward = kwargs.get("percent_reward", True)

        self.time_step = 5  # minutes
        self.run_duration = 14  # days
        self.steps_per_day = int(24 * 60 / self.time_step)
        self.total_steps = int(self.steps_per_day * self.run_duration)

        # Agent must keep light off outside this time range
        self.sim_start_hour = 9
        self.sim_end_hour = 21
        self.sim_steps_per_day = int(
            (self.sim_end_hour - self.sim_start_hour) * 60 / self.time_step
        )

        self.steps = 0
        self.start_time = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        self.min_ppfd = 50
        self.max_ppfd = 150
        self.optimal_ppfd = 100

        self.initial_size = 300  # mm^2
        self.area_record = defaultdict(float)
        self.last_action = 0
        self.dli = 0

        self.gamma = 0.99

    def get_tod(self):
        day = self.steps // self.steps_per_day
        minutes_today = (self.steps % self.steps_per_day) * self.time_step
        hour = minutes_today // 60
        minute = minutes_today % 60
        tod = self.start_time.replace(hour=hour, minute=minute) + datetime.timedelta(
            days=day
        )
        return tod

    def get_area(self, tod, action):
        # If morning, increase area by overnight growth
        if self.is_first_tod(tod):
            daily_optimal_ppfd = self.sim_steps_per_day * self.optimal_ppfd
            if self.dli <= daily_optimal_ppfd:
                percent_overnight_growth = 0.2 * (self.dli / daily_optimal_ppfd)
            else:
                percent_overnight_growth = 0.2 * (
                    (2 * daily_optimal_ppfd - self.dli) / daily_optimal_ppfd
                )
            self.current_morning_size += (
                self.current_morning_size * percent_overnight_growth
            )
            self.dli = 0  # Reset daily light integral

        if action == 0:
            area = 0
        else:
            steps_since_midnight = self.steps % self.steps_per_day
            steps_since_morning = (
                steps_since_midnight - self.sim_start_hour * 60 / self.time_step
            )
            area = self.current_morning_size * (
                1 + self.daily_area_curve(steps_since_morning)
            )
        return area

    def reward_function(self, tod):
        if self.is_first_tod(tod):
            yesterday_first_tod = tod - datetime.timedelta(days=1)
            current_area = self.area_record[tod]
            yesterday_area = self.area_record.get(yesterday_first_tod, 0.0)

            if yesterday_area == 0.0:
                return 0

            if self.percent_reward:
                return current_area / yesterday_area - 1
            else:
                return current_area - yesterday_area
        else:
            return 0

    async def start(self):
        self.steps = 0
        self.current_morning_size = self.initial_size
        self.area_record = defaultdict(float)
        action = 0
        self.dli = 0

        tod = self.get_tod()
        area = self.get_area(tod, action)
        self.area_record[tod] = area

        self.current_state = [tod, area]

        return self.current_state, self.get_info()

    async def step(self, action):
        # Make sure the agent behaves within constraints
        last_tod = self.get_tod()
        if self.is_night(last_tod):
            if action != 0:
                raise Exception("Your agent didn't turn off light at night.")
        elif action < self.min_ppfd or action > self.max_ppfd:
            raise Exception(
                f"Invalid action during daytime: {action} ppfd. Daytime action must be between {self.min_ppfd} and {self.max_ppfd} ppfd."
            )

        self.steps += 1
        self.dli += action

        tod = self.get_tod()
        area = self.get_area(tod, action)
        self.area_record[tod] = area

        self.current_state = [tod, area]
        reward = self.reward_function(tod)

        if self.steps == self.total_steps:
            return reward, self.current_state, True, self.get_info()
        else:
            return reward, self.current_state, False, self.get_info()

    def daily_area_curve(self, x):
        """
        Model the daily area curve by a Gaussian curve.
        Input: x is the number of steps so far today.
        """
        normalized_dli = self.dli / self.optimal_ppfd
        slowing = 1 + (x - normalized_dli) / self.sim_steps_per_day
        x = x / slowing

        stdev = self.sim_steps_per_day
        mu = self.sim_steps_per_day / 2
        gaussian = np.exp(-0.5 * ((x - mu) / stdev) ** 2)
        shift = np.exp(-0.5 * ((0 - mu) / stdev) ** 2)
        return gaussian - shift

    def is_night(self, tod) -> bool:
        is_night = tod.hour >= self.sim_end_hour or tod.hour < self.sim_start_hour
        return is_night

    def is_first_tod(self, tod) -> bool:
        is_first_tod = tod.hour == self.sim_start_hour and tod.minute == 5
        return is_first_tod

    def get_info(self):
        return {"gamma": self.gamma}
