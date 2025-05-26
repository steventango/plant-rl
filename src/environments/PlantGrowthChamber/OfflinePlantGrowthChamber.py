import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils.functions import normalize

logger = logging.getLogger("OfflinePlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class OfflinePlantGrowthChamber:
    def __init__(self, *args, **kwargs):
        self.dataset_paths = sorted(Path(path) for path in kwargs["dataset_paths"])
        self.dataset_index = 0
        self.index = 0
        self.normalize_reward = kwargs.get("normalize_reward", True)
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

        # Populate daily_mean_clean_areas
        local_dates = df['time'].dt.date
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
        normalized_mean_clean_area = normalize(mean_clean_area, 0, 50)
        clipped_mean_clean_area = np.clip(normalized_mean_clean_area, 0, 1)
        return clipped_seconds_since_morning, clipped_mean_clean_area

    def get_action(self):
        return self.dataset.iloc[self.index]["agent_action"]

    def get_reward(self):
        if self.index == 0:
            return 0.0
        if self.dataset.iloc[self.index]["time"].date() == self.dataset.iloc[self.index - 1]["time"].date():
            return 0.0

        current_local_date = self.dataset.iloc[self.index]["time"].date()
        yesterday_local_date = current_local_date - timedelta(days=1)

        if yesterday_local_date not in self.daily_mean_clean_areas:
            return 0.0

        current_morning_area = self.daily_mean_clean_areas.get(current_local_date, [0])[0]
        yesterday_morning_area = self.daily_mean_clean_areas.get(yesterday_local_date, [0])[0]

        if self.normalize_reward:
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
        terminal = self.index >= len(self.dataset) - 1
        if terminal:
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        return self.get_reward(), self.get_observation(), terminal, info
