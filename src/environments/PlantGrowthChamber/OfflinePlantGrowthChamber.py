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
    def __init__(self, *args, **kwargs):
        self.dataset_paths = sorted(Path(path) for path in kwargs["dataset_paths"])
        self.dataset_index = 0
        self.index = 0
        self.daily_area = kwargs.get("daily_area", True)
        self.daily_reward = kwargs.get("daily_reward", True)  
        self.use_photon_count = kwargs.get("use_photon_count", False)   # if true, "area" is replaced by "photon count" in the state
        self.daily_area_indicator = {}
        self.photon_counter = 0

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

        # Compute an area indicator for each day
        local_dates = df['time'].dt.date
        for date_val, group in df.groupby(local_dates):            
            self.daily_area_indicator[date_val] = np.mean(np.sort(group['mean_clean_area'])[-5:])  # max area  
            #self.daily_area_indicator[date_val] = np.mean(group['mean_clean_area'][:5])  # morning area (need to reset normalization bounds if using this)

        return df

    def get_observation(self):
        local_time = self.dataset.iloc[self.index]["time"]
        morning_time = local_time.replace(hour=9, minute=10, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)

        mean_clean_area = self.dataset.iloc[self.index]["mean_clean_area"]
        if self.daily_area:
            area_indicator = self.daily_area_indicator[pd.to_datetime(local_time).date()] 
            normalized_mean_clean_area = normalize(mean_clean_area / area_indicator, 0.75, 1.05)   # bounds suitable when using max area as benchmark
        else:
            normalized_mean_clean_area = normalize(mean_clean_area, 0, 100)   #TODO check if bounds appropriate

        if self.use_photon_count: 
            return np.clip(normalized_seconds_since_morning, 0, 1), np.clip(normalize(self.photon_counter, 0, 72), 0, 1)  # tile coder hash table can overflow without clipping
        else: 
            return np.clip(normalized_seconds_since_morning, 0, 1), np.clip(normalized_mean_clean_area, 0, 1)   

    def get_action(self):
        agent_action = self.dataset.iloc[self.index]['agent_action']
        return int(agent_action)

    def get_reward(self):
        if self.index == 0:
            return 0.0
        if self.dataset.iloc[self.index]["time"].date() == self.dataset.iloc[self.index - 1]["time"].date():
            return 0.0

        current_local_date = self.dataset.iloc[self.index]["time"].date()
        yesterday_local_date = current_local_date - timedelta(days=1)

        current_area_indicator = self.daily_area_indicator.get(current_local_date, 0)
        yesterday_area_indicator = self.daily_area_indicator.get(yesterday_local_date, 0)

        if self.daily_reward:
            reward = normalize(current_area_indicator / yesterday_area_indicator - 1, 0, 0.35)
        else:
            reward = normalize(current_area_indicator - yesterday_area_indicator, 0, 50)   # TODO: check if bounds appropriate

        return reward

    def start(self):
        self.dataset = self.load_dataset(self.dataset_paths[self.dataset_index]) 
        self.index = 0
        self.photon_counter = 0
        return self.get_observation(), {"action": self.get_action()}

    def step(self, _):
        # Update action history based on previous action
        self.photon_counter += self.get_action()

        # Compute next state
        self.index += 1 
        if self.dataset.iloc[self.index]["time"].date() != self.dataset.iloc[self.index - 1]["time"].date():
            self.photon_counter = 0
        obs = self.get_observation()

        # Compute reward
        r = self.get_reward()
          
        # Check if terminal
        terminal = self.index >= len(self.dataset) - 1
        info = {}
        if terminal:
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
        s0, s1 = super().get_observation()
        if self.use_photon_count: 
            return s0, np.clip(normalize(self.photon_counter, 0, 12), 0, 1)
        else: 
            return s0, s1

    def get_action(self):
        agent_actions = self.dataset.iloc[self.index:self.index + 6]['agent_action']
        average_action = agent_actions.sum()
        if average_action >= 4: 
            return int(1)
        elif average_action <= 2:
            return int(0)
        elif average_action == 3: 
            return int(np.random.choice([0, 1]))
    
    def step(self, _):
        # Update action history based on previous action
        self.photon_counter += self.get_action()

        # Compute next state
        self.index += 6
        if self.dataset.iloc[self.index]["time"].date() != self.dataset.iloc[self.index - 1]["time"].date():
            self.photon_counter = 0
        obs = self.get_observation()

        # Compute reward
        r = self.get_reward()
          
        # Check if terminal
        terminal = self.index + 6 >= len(self.dataset)
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
        current_local_date = self.dataset.iloc[self.index]["time"].date()
        current_area_indicator = self.daily_area_indicator.get(current_local_date)
        if self.dataset.iloc[self.index]["time"].date() == self.dataset.iloc[self.index - 1]["time"].date():
            tomorrow_local_date = current_local_date + timedelta(days=1)
            tomorrow_area_indicator = self.daily_area_indicator.get(tomorrow_local_date)
            reward = normalize(tomorrow_area_indicator / current_area_indicator - 1, 0, 0.35)
        else: 
            yesterday_local_date = current_local_date - timedelta(days=1)
            yesterday_area_indicator = self.daily_area_indicator.get(yesterday_local_date)
            reward = normalize(current_area_indicator / yesterday_area_indicator - 1, 0, 0.35)

        return reward
    
    def step(self, _):
        r, obs, terminal, info = super().step(_)

        # We need an earlier terminal here to have an extra day at the end for computing reward.
        # (maybe it's better to do this for the TD method above too)
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
    
class OfflinePlantGrowthChamber_1hrStep_MC_AreaOnly(OfflinePlantGrowthChamber_1hrStep_MC):   
    def get_observation(self):
        s0, s1 = super().get_observation()
        if self.use_photon_count: 
            raise ValueError(f'This env is incompatible with use_photon_count.')
        else: 
            return 0.0, s1

class OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly(OfflinePlantGrowthChamber_1hrStep_MC):   
    def get_observation(self):
        s0, s1 = super().get_observation()
        return s0, 0.0
        