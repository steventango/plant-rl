import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

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
        self.daily_area_indicator = {}


    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        processed_csv_paths = sorted(dataset_path.glob("raw.csv"))
        df = pd.read_csv(processed_csv_paths[-1])   # TODO: should it be -1 here or reflect self.dataset_index
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

        # Remove incomplete days
        df = self.remove_incomplete_days(df)

        # Compute an area indicator every day
        local_dates = df['time'].dt.date
        for date_val, group in df.groupby(local_dates):            
            self.daily_area_indicator[date_val] = np.mean(np.sort(group['mean_clean_area'])[-5:])  # max area  
            #self.daily_area_indicator[date_val] = np.mean(group['mean_clean_area'][:5])  # morning area  

        return df

    def get_observation(self):
        local_time = self.dataset.iloc[self.index]["time"]
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)

        mean_clean_area = self.dataset.iloc[self.index]["mean_clean_area"]
        if self.daily_area:
            area_indicator = self.daily_area_indicator[local_time.dt.date]  #TODO check if date syntax correct
            normalized_mean_clean_area = normalize(mean_clean_area / area_indicator, 0.75, 1.05)   # bounds suitable when using max area as benchmark
        else:
            normalized_mean_clean_area = normalize(mean_clean_area, 0, 100)   #TODO check if bounds appropriate

        return normalized_seconds_since_morning, normalized_mean_clean_area

    def get_action(self):
        agent_action = self.dataset.iloc[self.index]["agent_action"].astype(int)
        if agent_action < 0 or agent_action > 3:
            agent_action = -1
        return agent_action

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
        self.dataset = self.load_dataset(self.dataset_paths[self.dataset_index])  # TODO: check if syntax correct here
        self.index = 0
        return self.get_observation(), {"action": self.get_action()}

    def step(self, _):
        self.index += 1
        info = {"action": self.get_action()}
        terminal = self.index >= len(self.dataset) - 1
        if terminal:
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        return self.get_reward(), self.get_observation(), terminal, info
    
    def remove_incomplete_days(self, df, min_timestamps = 72):  # Note on min_timestamps: Exp 3 has 73 time stamps per day, but older datasets may have only 72.
        local_dates = df['time'].dt.date
        complete_dates = []
        for date_val, group in df.groupby(local_dates):            
            if len(group) >= min_timestamps:   
                complete_dates.append(date_val)

        return df[df['time'].dt.date.isin(complete_dates)]