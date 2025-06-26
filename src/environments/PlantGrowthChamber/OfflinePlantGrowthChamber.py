import logging  # type: ignore
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from utils.functions import normalize

logger = logging.getLogger("OfflinePlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class OfflinePlantGrowthChamber:
    """
    Use Exp 3 data to do offline learning.
    State = [time of day,
             daily light integral (dli),
             plant area,
             plant openness] (all normalized to [0,1])
    Reward = percentage or raw overnight growth assigned to last time stamp of each day
    """

    def __init__(self, *args, **kwargs):
        self.dataset_paths = sorted(Path(path) for path in kwargs["dataset_paths"])
        self.dataset_index = 0
        self.index = 0
        self.daily_reward = kwargs.get(
            "daily_reward", True
        )  # if true, use "percentage" overnight growth as reward
        self.reward_type = kwargs.get("reward_type", "max_mean")
        self.daily_morning_areas = {}
        self.daily_max_areas = {}
        self.daily_areas = {}
        self.dli = 0

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        processed_csv_paths = sorted(dataset_path.glob("raw.csv"))
        df = pd.read_csv(processed_csv_paths[-1])
        df["time"] = pd.to_datetime(df["time"])
        edmonton_tz = pytz.timezone("America/Edmonton")
        df["time"] = df["time"].dt.tz_convert(edmonton_tz)

        df = (
            df.groupby("time")
            .agg(
                {
                    "plant_id": "first",
                    "action.0": "first",
                    "mean_clean_area": "first",
                }
            )
            .reset_index()
        )

        # Remove time stamps before 9:30 and after 20:30, which are not relevant when twilight is used
        df = df[
            (time(9, 30) <= df["time"].dt.time) & (df["time"].dt.time <= time(20, 31))
        ]

        # Remove incomplete days
        df = self.remove_incomplete_days(df, timestamps_per_day=67)

        # Compute morning and max areas on each day
        local_dates = df["time"].dt.date  # type: ignore
        for date_val, group in df.groupby(local_dates):  # type: ignore
            self.daily_morning_areas[date_val] = np.mean(
                group["mean_clean_area"][:5]
            )  # morning areas
            self.daily_max_areas[date_val] = np.mean(
                np.sort(group["mean_clean_area"])[-10:]
            )  # max areas
            self.daily_areas[date_val] = group["mean_clean_area"].values  # type: ignore

        return df  # type: ignore

    def get_observation(self):
        # tod
        local_time = self.dataset.iloc[self.index - 1]["time"]
        morning_time = local_time.replace(hour=9, minute=30, second=0, microsecond=0)
        seconds_since_morning = (local_time - morning_time).total_seconds()
        normalized_seconds_since_morning = seconds_since_morning / (11 * 3600)

        # area
        mean_clean_area = self.dataset.iloc[self.index - 1]["mean_clean_area"]
        normalized_mean_clean_area = normalize(mean_clean_area, 0, 680)

        # openness
        morning_area = self.daily_morning_areas[pd.to_datetime(local_time).date()]
        normalized_openness = normalize(mean_clean_area / morning_area, 0.93, 1.3)

        observation = np.hstack(
            [
                np.clip(normalized_seconds_since_morning, 0, 1),
                np.clip(normalize(self.dli, 0, 66), 0, 1),
                np.clip(normalized_mean_clean_area, 0, 0.9999),
                np.clip(normalized_openness, 0, 0.9999),
            ]
        )
        return observation

    def get_action(self):
        agent_action = self.dataset.iloc[self.index]["action.0"] > 0.3
        return int(agent_action)

    def get_reward(self):
        if self.index == 0:
            return 0.0
        if self.daily_reward:
            if (
                self.dataset.iloc[self.index]["time"].date()
                == self.dataset.iloc[self.index - 1]["time"].date()
            ):
                return 0.0

            today_local_date = self.dataset.iloc[self.index]["time"].date()
            yesterday_local_date = today_local_date - timedelta(days=1)

            if self.reward_type == "max_mean":
                today_areas = self.daily_areas.get(today_local_date, [])
                today_area = (
                    np.mean(np.sort(today_areas)[-10:]) if len(today_areas) else 0
                )
                yesterday_areas = self.daily_areas.get(yesterday_local_date, [])
                yesterday_area = (
                    np.mean(np.sort(yesterday_areas)[-10:])
                    if len(yesterday_areas)
                    else 0
                )
            elif self.reward_type == "max":
                today_areas = self.daily_areas.get(today_local_date, [])
                today_area = np.max(today_areas) if len(today_areas) else 0
                yesterday_areas = self.daily_areas.get(yesterday_local_date, [])
                yesterday_area = np.max(yesterday_areas) if len(yesterday_areas) else 0
            elif self.reward_type == "mean":
                today_areas = self.daily_areas.get(today_local_date, [])
                today_area = np.mean(today_areas) if len(today_areas) else 0
                yesterday_areas = self.daily_areas.get(yesterday_local_date, [])
                yesterday_area = np.mean(yesterday_areas) if len(yesterday_areas) else 0
            elif self.reward_type == "first":
                today_areas = self.daily_areas.get(today_local_date, [])
                today_area = today_areas[0] if len(today_areas) else 0
                yesterday_areas = self.daily_areas.get(yesterday_local_date, [])
                yesterday_area = yesterday_areas[0] if len(yesterday_areas) else 0
            else:
                raise ValueError(f"Invalid reward type: {self.reward_type}")
            if yesterday_area == 0:
                reward = 0.0
            else:
                reward = normalize(today_area / yesterday_area - 1, 0, 0.35)
        else:
            current_area = self.dataset.iloc[self.index]["mean_clean_area"]
            previous_area = self.dataset.iloc[self.index - 1]["mean_clean_area"]
            reward = normalize(current_area - previous_area, 0, 150)

        return reward

    def get_light_amount(self, action):
        if action == 1:
            return 1.0
        else:
            return 0.5

    def start(self):
        self.dataset = self.load_dataset(self.dataset_paths[self.dataset_index])
        self.index = 1
        self.dli = 0
        return self.get_observation(), {"action": self.get_action()}

    def step(self, _):
        # Update action history based on previous action
        self.dli += self.get_light_amount(self.get_action())

        # Compute next state
        self.index += 1
        if (
            self.dataset.iloc[self.index]["time"].date()
            != self.dataset.iloc[self.index - 1]["time"].date()
        ):
            self.dli = 0
        obs = self.get_observation()

        # Compute reward
        r = self.get_reward()

        # Check if terminal
        terminal = self.index + 72 >= len(self.dataset)
        info = {}
        if terminal:
            dataset_path = self.dataset_paths[self.dataset_index]
            logger.info(
                f"Added {self.index} transitions from {dataset_path} to the replay buffer."
            )
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        else:
            info.update({"action": self.get_action()})

        return r, obs, terminal, info

    def remove_incomplete_days(self, df, timestamps_per_day=72):
        local_dates = df["time"].dt.date
        complete_dates = []
        for date_val, group in df.groupby(local_dates):
            if len(group) == timestamps_per_day:
                complete_dates.append(date_val)

        return df[df["time"].dt.date.isin(complete_dates)]


class OfflinePlantGrowthChamberTime(OfflinePlantGrowthChamber):
    def get_observation(self):
        super_obs = super().get_observation()
        obs = super_obs[[0]]
        return obs


class OfflinePlantGrowthChamberTimeDLI(OfflinePlantGrowthChamber):
    def get_observation(self):
        super_obs = super().get_observation()
        obs = super_obs[[0, 1]]
        return obs


class OfflinePlantGrowthChamberTimeArea(OfflinePlantGrowthChamber):
    def get_observation(self):
        super_obs = super().get_observation()
        obs = super_obs[[0, 2]]
        return obs


class OfflinePlantGrowthChamber_1hrStep(OfflinePlantGrowthChamber):
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[0], np.clip(normalize(self.dli, 0, 12), 0, 1), s[2], s[3]])

    def get_action(self):
        agent_actions = (
            self.dataset.iloc[self.index : self.index + 6]["action.0"] > 0.3
        ).astype(int)
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
        if (
            self.dataset.iloc[self.index]["time"].date()
            != self.dataset.iloc[self.index - 1]["time"].date()
        ):
            self.dli = 0
        obs = self.get_observation()

        # Compute reward
        r = self.get_reward()

        # Check if terminal
        terminal = self.index + 72 >= len(self.dataset)
        info = {}
        if terminal:
            logger.info(
                f"Added {int(self.index / 6)} transitions to the replay buffer."
            )
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset_paths):
                info.update({"exhausted": True})
        else:
            info.update({"action": self.get_action()})

        return r, obs, terminal, info


class OfflinePlantGrowthChamber_1hrStep_MC(OfflinePlantGrowthChamber_1hrStep):
    """Assign the same overnight-growth reward to all steps during that day"""

    def get_reward(self):
        today_local_date = self.dataset.iloc[self.index]["time"].date()
        today_max_area = self.daily_max_areas.get(today_local_date)
        if (
            self.dataset.iloc[self.index]["time"].date()
            == self.dataset.iloc[self.index - 1]["time"].date()
        ):
            tomorrow_local_date = today_local_date + timedelta(days=1)
            tomorrow_max_area = self.daily_max_areas.get(tomorrow_local_date)
            if self.daily_reward:
                reward = normalize(tomorrow_max_area / today_max_area - 1, 0, 0.35)  # type: ignore
            else:
                reward = normalize(tomorrow_max_area - today_max_area, 0, 150)  # type: ignore
        else:
            yesterday_local_date = today_local_date - timedelta(days=1)
            yesterday_max_area = self.daily_max_areas.get(yesterday_local_date)
            if self.daily_reward:
                reward = normalize(today_max_area / yesterday_max_area - 1, 0, 0.35)  # type: ignore
            else:
                reward = normalize(today_max_area - yesterday_max_area, 0, 150)  # type: ignore

        return reward


class OfflinePlantGrowthChamber_1hrStep_MC_OpennessOnly(
    OfflinePlantGrowthChamber_1hrStep_MC
):
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[3], 0.0])


class OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly(
    OfflinePlantGrowthChamber_1hrStep_MC
):
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[0], 0.0])


class OfflinePlantGrowthChamber_1hrStep_MC_Area_Openness(
    OfflinePlantGrowthChamber_1hrStep_MC
):
    def get_observation(self):
        s = super().get_observation()
        return np.hstack([s[2], s[3]])
