import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("OfflinePlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class OfflinePlantGrowthChamber:
    def __init__(self, *args, **kwargs):
        self.dataset_paths = iter(Path(path) for path in kwargs["dataset_paths"])
        self.index = 0

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        processed_csv_paths = sorted(dataset_path.glob("processed/**/all.csv"))
        df = pd.read_csv(processed_csv_paths[-1])
        df["time"] = pd.to_datetime(df["time"])
        df = (
            df.groupby("time")
            .agg(
                {
                    "plant_id": "first",
                    "agent_action": "first",
                    "time": "first",
                    "clean_area": "mean",
                }
            )
            .reset_index()
        )
        return df

    def get_observation(self):
        return (self.dataset.iloc[self.index]["time"], self.dataset.iloc[self.index]["clean_area"])

    def get_action(self):
        return self.dataset.iloc[self.index]["agent_action"]

    def get_reward(self):
        return self.dataset.iloc[self.index]["clean_area"] - self.dataset.iloc[self.index - 1]["clean_area"] if self.index > 0 else 0

    def start(self):
        self.dataset = self.load_dataset(next(self.dataset_paths))
        self.index = 0
        return self.get_observation(), {"action": self.get_action()}

    def step(self):
        self.index += 1
        terminal = self.index >= len(self.dataset) - 1
        return self.get_reward(), self.get_observation(), terminal, {"action": self.get_action()}
