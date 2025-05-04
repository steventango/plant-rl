import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from environments.PlantGrowthChamber.zones import deserialize_zone


class MockPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, *args, **kwargs):
        self.dataset_path = Path(kwargs["dataset_path"])
        self.config_path = self.dataset_path / "config.json"
        self.dataset_df = pd.read_csv(self.dataset_path / "raw.csv")
        self.dataset_df["time"] = pd.to_datetime(self.dataset_df["time"])
        with open(self.config_path) as f:
            self.config = json.load(f)
        kwargs["zone"] = deserialize_zone(self.config["zone"])
        self.index = 0
        super().__init__(*args, **kwargs)

        self.images_path = self.dataset_path / "images"

    async def start(self):
        result = await super().start()
        return result

    async def get_image(self):
        row = self.dataset_df.iloc[self.index]
        path = self.images_path / row["image_name"]
        self.images["left"] = Image.open(path)

    def get_time(self):
        row = self.dataset_df.iloc[self.index]
        return row["time"]

    def get_terminal(self):
        return self.index >= len(self.dataset_df) - 1

    async def put_action(self, action: int):
        self.last_action = action

    async def sleep_until(self, wake_time: datetime):
        while self.get_time() < wake_time:
            self.index += 1

    async def close(self):
        pass
