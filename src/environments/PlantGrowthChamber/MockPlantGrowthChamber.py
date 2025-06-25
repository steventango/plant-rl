import json
import logging
from datetime import datetime
from pathlib import Path
# from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from environments.PlantGrowthChamber.zones import deserialize_zone

logger = logging.getLogger("MockPlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class MockPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        self.dataset_path = Path(kwargs["dataset_path"])
        self.config_path = self.dataset_path / "config.json"
        self.procesed_csv_paths = sorted(self.dataset_path.glob("processed/**/all.csv"))
        if len(self.procesed_csv_paths) == 0:
            self.dataset_df = pd.read_csv(self.dataset_path / "raw.csv")
        else:
            self.dataset_df = pd.read_csv(self.procesed_csv_paths[-1])
        self.dataset_df["time"] = pd.to_datetime(self.dataset_df["time"])
        with open(self.config_path) as f:
            self.config = json.load(f)
        kwargs["zone"] = deserialize_zone(self.config["zone"])
        self.index = 0
        self.time = self.dataset_df["time"].min()
        # For testing purposes, we can set a fixed time
        # tzinfo = ZoneInfo("America/Edmonton")
        # self.time = datetime(2022, 7, 22, 9, 25, 0, tzinfo=tzinfo)
        self.mock_area = kwargs.get("mock_area", False)
        self.plant_stat_columns = [
            "in_bounds",
            "area",
            "convex_hull_area",
            "solidity",
            "perimeter",
            "width",
            "height",
            "longest_path",
            "center_of_mass_x",
            "center_of_mass_y",
            "convex_hull_vertices",
            "object_in_frame",
            "ellipse_center_x",
            "ellipse_center_y",
            "ellipse_major_axis",
            "ellipse_minor_axis",
            "ellipse_angle",
            "ellipse_eccentricity",
        ]
        super().__init__(*args, **kwargs)

        self.images_path = self.dataset_path / "images"

        self.simulate = kwargs.get("simulate", False)
        self.dim_action_penalty = 0
        self.awoken = False

    async def start(self):
        result = await super().start()
        return result

    async def get_image(self):
        row = self.dataset_df[self.dataset_df["frame"] == self.index].iloc[0]
        path = self.images_path / row["image_name"]
        self.images["left"] = Image.open(path)

    def get_time(self):
        return self.time

    def get_terminal(self):
        return self.index >= self.dataset_df["frame"].max()

    def get_plant_stats(self):
        if self.mock_area:
            self.df = self.dataset_df[self.dataset_df["frame"] == self.index]
            self.df = self.df[self.plant_stat_columns]
            self.plant_stats = np.array(self.df, dtype=np.float32)
        else:
            super().get_plant_stats()

    async def step(self, action: np.ndarray):
        reward, observation, terminal, info = await super().step(action)
        if self.simulate and np.sum(action) == 0:
            reward -= 0.5 * abs(reward)
        return reward, observation, terminal, info

    async def put_action(self, action: int):
        logger.debug(f"action: {action}")
        self.last_action = action

    async def sleep_until(self, wake_time: datetime):
        while self.time < wake_time and not self.get_terminal():
            self.time = wake_time
            if self.index + 1 >= self.dataset_df["frame"].max():
                self.index = 0
                break
            next_row = self.dataset_df[self.dataset_df["frame"] == self.index + 1].iloc[
                0
            ]
            next_time = next_row["time"]
            if next_time <= self.time:
                self.time = next_time
                self.index += 1

    async def close(self):
        pass
