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
        self.daily_photon = kwargs.get("use_photon_count", False)   # if true, "area" is replaced by "photon count" in the state
        self.daily_area = kwargs.get("daily_area", True)
        self.daily_reward = kwargs.get("daily_reward", True)  
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
                    "episode": "first"
                }
            )
            .reset_index()
        )

        # Remove incomplete episodes
        df = self.remove_incomplete_days(df)

        # Remove time stamps at 9:05, which was added for the daily episode scenario
        df = df[df['time'].dt.strftime('%H:%M') != '09:05']

        # Compute an area indicator every day
        local_dates = df['time'].dt.date
        for date_val, group in df.groupby(local_dates):            
            self.daily_area_indicator[date_val] = np.mean(np.sort(group['mean_clean_area'])[-5:])  # max area  
            #self.daily_area_indicator[date_val] = np.mean(group['mean_clean_area'][:5])  # morning area (you have to reset normalization bounds if using this)

        return df

    def get_observation(self):
        local_time = self.dataset.iloc[self.index]["time"]
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)

        mean_clean_area = self.dataset.iloc[self.index]["mean_clean_area"]
        if self.daily_area:
            area_indicator = self.daily_area_indicator[pd.to_datetime(local_time).date()] 
            normalized_mean_clean_area = normalize(mean_clean_area / area_indicator, 0.75, 1.05)   # bounds suitable when using max area as benchmark
        else:
            normalized_mean_clean_area = normalize(mean_clean_area, 0, 100)   #TODO check if bounds appropriate

        if not self.daily_photon: 
            return normalized_seconds_since_morning, normalized_mean_clean_area
        else: 
            return normalized_seconds_since_morning, normalize(self.photon_counter, 0, 72)

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

        if yesterday_local_date not in self.daily_area_indicator:   
            return 0.0

        current_area_indicator = self.daily_area_indicator.get(current_local_date, 0)
        yesterday_area_indicator = self.daily_area_indicator.get(yesterday_local_date, 0)

        if self.daily_reward:
            if yesterday_area_indicator == 0:  # Avoid division by zero
                return 0.0
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
            # Compute action taken at next state
            info.update({"action": self.get_action()})
        return r, obs, terminal, info
    
    def remove_incomplete_days(self, df, timestamps_per_day = 73):  # Exp 3 has 73 time stamps per day.
        local_dates = df['time'].dt.date
        complete_dates = []
        incomplete_dates = []
        
        # Identify complete and incomplete dates
        for date_val, group in df.groupby(local_dates):            
            if len(group) == timestamps_per_day:   
                complete_dates.append(date_val)
            else:
                incomplete_dates.append(date_val)
        
        # Start with all rows from complete dates
        result_df = df[df['time'].dt.date.isin(complete_dates)]
        
        # For each incomplete date, check if previous day is complete
        for incomplete_date in incomplete_dates:
            prev_date = incomplete_date - pd.Timedelta(days=1)
            
            if prev_date in complete_dates:
                # Keep only 9:10am entry from this incomplete date
                incomplete_day_rows = df[df['time'].dt.date == incomplete_date]
                am_910_rows = incomplete_day_rows[incomplete_day_rows['time'].dt.strftime('%H:%M') == '09:10']
                result_df = pd.concat([result_df, am_910_rows])
        
        return result_df.sort_values('time').reset_index(drop=True)
    
class OfflinePlantGrowthChamber_1hrStep(OfflinePlantGrowthChamber):
    def get_observation(self):
        tod = floor((self.index % 72) / 6)
        normalized_tod = normalize(tod, 0, 12)

        if not self.daily_photon: 
            raise ValueError('Daily area has not been implemented. Please set daily_photon=true.')
        else: 
            return normalized_tod, normalize(self.photon_counter, 0, 12)

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
        terminal = self.index >= len(self.dataset) - 1
        info = {}
        if terminal:
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        else: 
            # Compute action taken at next state
            info.update({"action": self.get_action()})
        return r, obs, terminal, info