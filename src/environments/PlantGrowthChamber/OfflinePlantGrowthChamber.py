import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils.functions import normalize
from environments.PlantGrowthChamber.utils import get_one_hot_time_observation

logger = logging.getLogger("OfflinePlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class OfflinePlantGrowthChamber:
    def __init__(self, *args, **kwargs):
        self.dataset_paths = [Path(path) for path in kwargs["dataset_paths"]]
        self.dataset_index = 0
        self.index = 0
        self.daily_area = kwargs.get("daily_area", False)
        self.daily_reward = kwargs.get("daily_reward", True)
        self.daily_mean_clean_areas = {}

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        processed_csv_paths = sorted(dataset_path.glob("raw.csv"))
        df = pd.read_csv(processed_csv_paths[-1])
        df["time"] = pd.to_datetime(df["time"])
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

        df = self.remove_incomplete_days(df)

        # Populate daily_mean_clean_areas
        local_dates = df['time'].dt.date
        df['daily_action_sum'] = df.groupby(local_dates)['agent_action'].cumsum()
        for date_val, group in df.groupby(local_dates):
            self.daily_mean_clean_areas[date_val] = group['mean_clean_area'].tolist()

        return df

    def get_observation(self):
        utc_time = self.dataset.iloc[self.index]["time"]
        local_time = utc_time.tz_convert("America/Edmonton")
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)
        clipped_seconds_since_morning = np.clip(normalized_seconds_since_morning, 0, 1)
        mean_clean_area = self.dataset.iloc[self.index]["mean_clean_area"]
        if self.daily_area:
            current_local_date = self.dataset.iloc[self.index]["time"].date()
            current_morning_area = self.daily_mean_clean_areas.get(current_local_date, [0])[0]
            normalized_mean_clean_area = (mean_clean_area - current_morning_area) / current_morning_area
        else:
            normalized_mean_clean_area = normalize(mean_clean_area, 0, 100)
        # clipped_mean_clean_area = np.clip(normalized_mean_clean_area, 0, 1)
        # return clipped_seconds_since_morning, 0.5
        return get_one_hot_time_observation(local_time)

    def get_action(self):
        agent_action = self.dataset.iloc[self.index]["agent_action"]
        if pd.isna(agent_action):
            agent_action = -1
        else:
            agent_action = int(agent_action)

        return agent_action

    def get_reward(self):
        if self.index == 0:
            return 0.0
        if self.dataset.iloc[self.index]["time"].tz_convert("America/Edmonton").date() == self.dataset.iloc[self.index - 1]["time"].tz_convert("America/Edmonton").date():
            return 0.0

        current_local_date = self.dataset.iloc[self.index]["time"].tz_convert("America/Edmonton").date()
        yesterday_local_date = current_local_date - timedelta(days=1)

        if yesterday_local_date not in self.daily_mean_clean_areas:
            return 0.0

        current_morning_area = self.daily_mean_clean_areas.get(current_local_date, [0])[0]
        yesterday_morning_area = self.daily_mean_clean_areas.get(yesterday_local_date, [0])[0]

        if self.daily_reward:
            if yesterday_morning_area == 0:  # Avoid division by zero
                return 0.0
            reward = normalize(current_morning_area / yesterday_morning_area - 1, 0, 0.35)
        else:
            reward = normalize(current_morning_area - yesterday_morning_area, 0, 50)

        return reward

    def start(self):
        self.dataset = self.load_dataset(self.dataset_paths[self.dataset_index])
        self.index = 0
        return self.get_observation(), {"action": self.get_action()}

    def step(self, _):
        self.index += 1
        info = {"action": self.get_action()}
        if self.index >= len(self.dataset) - 1:
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        return self.get_reward(), self.get_observation(), False, info

    def remove_incomplete_days(self, df, min_timestamps = 72):  # Note on min_timestamps: Exp 3 has 73 time stamps per day, but older datasets may have only 72.
        local_dates = df["time"].dt.tz_convert("America/Edmonton").dt.date
        complete_dates = []
        for date_val, group in df.groupby(local_dates):
            if len(group) >= min_timestamps:
                complete_dates.append(date_val)

        return df[local_dates.isin(complete_dates)]

class OfflinePlantGrowthChamberTOD(OfflinePlantGrowthChamber):
    def get_observation(self):
        utc_time = self.dataset.iloc[self.index]["time"]
        local_time = utc_time.tz_convert("America/Edmonton")
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)
        clipped_seconds_since_morning = np.clip(normalized_seconds_since_morning, 0, 1)
        return clipped_seconds_since_morning,

class OfflinePlantGrowthChamberSumLight(OfflinePlantGrowthChamber):
    def get_observation(self):
        utc_time = self.dataset.iloc[self.index]["time"]
        local_time = utc_time.tz_convert("America/Edmonton")
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (12 * 3600)
        clipped_seconds_since_morning = np.clip(normalized_seconds_since_morning, 0, 1)
        sum_prev_actions = self.dataset.iloc[self.index]["daily_action_sum"]
        return clipped_seconds_since_morning, sum_prev_actions
