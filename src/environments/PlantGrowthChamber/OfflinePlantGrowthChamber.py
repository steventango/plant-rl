import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from math import floor

from utils.functions import normalize

logger = logging.getLogger("OfflinePlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class OfflinePlantGrowthChamber:
    '''
    Use Exp 3 data to do offline learning.
    State = [time of day, 
             daily light integral (dli),
             plant area, 
             plant openness] (all normalized to [0,1])
    Reward = percentage or raw overnight growth assigned to last time stamp of each day 
    '''
    def __init__(self, *args, **kwargs):
        self.dataset_paths = sorted(Path(path) for path in kwargs["dataset_paths"])
        self.dataset_index = 0
        self.index = 0
        self.daily_reward = kwargs.get("daily_reward", True)   # if true, use "percentage" overnight growth as reward
        self.daily_morning_areas = {}
        self.daily_max_areas = {}
        self.dli = 0

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        processed_csv_paths = sorted(dataset_path.glob("raw.csv"))
        df = pd.read_csv(processed_csv_paths[-1])
        df["time"] = pd.to_datetime(df["time"])
        edmonton_tz = pytz.timezone('America/Edmonton')
        df['time'] = df['time'].dt.tz_convert(edmonton_tz)

        df = (
            df.groupby("time")
            .agg(
                {
                    "plant_id": "first",
                    "agent_action": "first",
                    "mean_clean_area": "first",
                }
            )
            .reset_index()
        )

        # Remove time stamps at 9:05, which was the terminal state added for the daily episode scenario
        df = df[df['time'].dt.strftime('%H:%M') != '09:05']

        # Remove incomplete days 
        df = self.remove_incomplete_days(df, timestamps_per_day = 72)

        # Compute morning and max areas on each day
        local_dates = df['time'].dt.date
        for date_val, group in df.groupby(local_dates):            
            self.daily_morning_areas[date_val] = np.mean(group['mean_clean_area'][:5])          # morning areas
            self.daily_max_areas[date_val] = np.mean(np.sort(group['mean_clean_area'])[-10:])   # max areas

        return df

    def get_observation(self):
        # tod
        local_time = self.dataset.iloc[self.index]["time"]
        morning_time = local_time.replace(hour=9, minute=10, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)

        # area
        mean_clean_area = self.dataset.iloc[self.index]["mean_clean_area"]
        normalized_mean_clean_area = normalize(mean_clean_area, 0, 680)

        # openness
        morning_area = self.daily_morning_areas[pd.to_datetime(local_time).date()] 
        normalized_openness = normalize(mean_clean_area / morning_area, 0.93, 1.3)

    
        observation = np.hstack([np.clip(normalized_seconds_since_morning, 0, 1),
                                 np.clip(normalize(self.dli, 0, 72), 0, 1),
                                 np.clip(normalized_mean_clean_area, 0, 0.9999),
                                 np.clip(normalized_openness, 0, 0.9999)
                                 ])
        return observation

    def get_action(self):
        agent_action = self.dataset.iloc[self.index]['agent_action']
        return int(agent_action)

    def get_reward(self):
        if self.index == 0:
            return 0.0
        if self.dataset.iloc[self.index]["time"].date() == self.dataset.iloc[self.index - 1]["time"].date():
            return 0.0

        today_local_date = self.dataset.iloc[self.index]["time"].date()
        yesterday_local_date = today_local_date - timedelta(days=1)

        today_max_area = self.daily_max_areas.get(today_local_date, 0)
        yesterday_max_area = self.daily_max_areas.get(yesterday_local_date, 0)
        if self.daily_reward:
            reward = normalize(today_max_area / yesterday_max_area - 1, 0, 0.35)
        else:
            reward = normalize(today_max_area - yesterday_max_area, 0, 150)
            
        return reward
    
    def get_light_amount(self, action):
        if action == 1:
            return 1.0
        else:   
            return 0.5

    def start(self):
        self.dataset = self.load_dataset(self.dataset_paths[self.dataset_index]) 
        self.index = 0
        self.dli = 0
        return self.get_observation(), {"action": self.get_action()}

    def step(self, _):
        # Update action history based on previous action
        self.dli += self.get_light_amount(self.get_action())

        # Compute next state
        self.index += 1 
        if self.dataset.iloc[self.index]["time"].date() != self.dataset.iloc[self.index - 1]["time"].date():
            self.dli = 0
        obs = self.get_observation()

        # Compute reward
        r = self.get_reward()
          
        # Check if terminal
        terminal = self.index + 72 >= len(self.dataset) 
        info = {}
        if terminal:
            logger.info(f'Added {self.index} transitions to the replay buffer.')
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        else: 
            info.update({"action": self.get_action()})

        return r, obs, terminal, info
    
    def remove_incomplete_days(self, df, timestamps_per_day = 72): 
        local_dates = df['time'].dt.date
        complete_dates = []
        for date_val, group in df.groupby(local_dates):            
            if len(group) == timestamps_per_day:   
                complete_dates.append(date_val)

        return df[df['time'].dt.date.isin(complete_dates)]
    
class OfflinePlantGrowthChamber_1hrStep(OfflinePlantGrowthChamber):
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[0], np.clip(normalize(self.dli, 0, 12), 0, 1), s[2], s[3]])
    
    def get_action(self):
        agent_actions = self.dataset.iloc[self.index:self.index + 6]['agent_action']
        average_action = agent_actions.sum()
        if average_action >= 4: 
            return int(1)
        else:
            return int(0)
    
    def step(self, _):
        # Update action history based on previous action
        self.dli += self.get_light_amount(self.get_action())

        # Compute next state
        self.index += 6
        if self.dataset.iloc[self.index]["time"].date() != self.dataset.iloc[self.index - 1]["time"].date():
            self.dli = 0
        obs = self.get_observation()

        # Compute reward
        r = self.get_reward()
          
        # Check if terminal
        terminal = self.index + 72 >= len(self.dataset)
        info = {}
        if terminal:
            logger.info(f'Added {int(self.index/6)} transitions to the replay buffer.')
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        else: 
            info.update({"action": self.get_action()})

        return r, obs, terminal, info
    
class OfflinePlantGrowthChamber_1hrStep_MC(OfflinePlantGrowthChamber_1hrStep):   
    '''Assign the same overnight-growth reward to all steps during that day'''

    def get_reward(self):
        today_local_date = self.dataset.iloc[self.index]["time"].date()
        today_max_area = self.daily_max_areas.get(today_local_date)
        if self.dataset.iloc[self.index]["time"].date() == self.dataset.iloc[self.index - 1]["time"].date():
            tomorrow_local_date = today_local_date + timedelta(days=1)
            tomorrow_max_area = self.daily_max_areas.get(tomorrow_local_date)
            if self.daily_reward:
                reward = normalize(tomorrow_max_area / today_max_area - 1, 0, 0.35)
            else:
                reward = normalize(tomorrow_max_area - today_max_area, 0, 150)
        else: 
            yesterday_local_date = today_local_date - timedelta(days=1)
            yesterday_max_area = self.daily_max_areas.get(yesterday_local_date)
            if self.daily_reward:
                reward = normalize(today_max_area / yesterday_max_area - 1, 0, 0.35)
            else: 
                reward = normalize(today_max_area - yesterday_max_area, 0, 150)

        return reward
    
class OfflinePlantGrowthChamber_1hrStep_MC_AreaOnly(OfflinePlantGrowthChamber_1hrStep_MC):   
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[2], 0.0])

class OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly(OfflinePlantGrowthChamber_1hrStep_MC):   
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[0], 0.0])