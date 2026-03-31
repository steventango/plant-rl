import glob
import json  # type: ignore
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from environments.PlantGrowthChamber.zones import deserialize_zone

logger = logging.getLogger("plant_rl.MockPlantGrowthChamber")


class MockPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        self.dataset_path = Path(
            kwargs.get("dataset_path", "/data/plant-rl/offline/v23/mixed-v23.parquet")
        )

        if self.dataset_path.is_dir():
            self.dataset_df = pl.read_parquet(self.dataset_path / "mixed-v23.parquet")
        else:
            self.dataset_df = pl.read_parquet(self.dataset_path)

        self.mock_stats = kwargs.get("mock_stats", kwargs.get("mock_area", False))

        # Filtering
        if "experiment" in kwargs:
            self.dataset_df = self.dataset_df.filter(
                pl.col("experiment") == kwargs["experiment"]
            )
        if "zone_id" in kwargs:
            self.dataset_df = self.dataset_df.filter(
                pl.col("zone") == kwargs["zone_id"]
            )

        # Sort and get unique wall times
        self.dataset_df = self.dataset_df.sort("wall_time")
        self.unique_wall_times = (
            self.dataset_df["wall_time"].unique(maintain_order=True).to_list()
        )
        self.time_index = 0
        self.wall_time = (
            self.unique_wall_times[self.time_index] if self.unique_wall_times else 0.0
        )

        # Pull initial time from the dataset if available
        if "time" in self.dataset_df.columns:
            self.current_time = self.dataset_df.filter(
                pl.col("wall_time") == self.wall_time
            )["time"][0]
        else:
            self.current_time = datetime.fromtimestamp(self.wall_time, tz=timezone.utc)

        # We still need some config for zones if not provided
        self.config_path = self.dataset_path.parent / "config.json"
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = json.load(f)
            if "zone" not in kwargs or kwargs["zone"] is None:
                kwargs["zone"] = deserialize_zone(self.config["zone"])

        super().__init__(*args, **kwargs)

        # Image resolution base path
        self.images_base_path = Path("/data/plant-rl/online")

        self.dim_action_penalty = 0
        self.awoken = False

    async def start(self):
        result = await super().start()
        return result

    async def get_image(self):
        if self.mock_stats:
            return

        row = self.dataset_df.filter(pl.col("wall_time") == self.wall_time).head(1)
        if row.height == 0:
            return

        image_name = row["image_name"][0]
        exp = row["experiment"][0]
        zone = row["zone"][0]

        # Common patterns for image paths
        path = (
            self.images_base_path
            / f"E{exp}"
            / "P1"
            / f"*{zone}"
            / f"alliance-zone{zone:02d}"
            / "images"
            / image_name
        )
        glob_path = str(path)

        matches = glob.glob(glob_path)
        if len(matches) == 0:
            logger.warning(f"Could not find image {image_name} in {path}")
        elif len(matches) > 1:
            logger.warning(f"Found multiple images {image_name} in {path}")

        self.images["left"] = Image.open(matches[0])

    def get_time(self):  # type: ignore
        return self.current_time

    def get_terminal(self):
        return self.time_index >= len(self.unique_wall_times) - 3

    async def get_plant_stats(self):
        if self.mock_stats:
            df_step = self.dataset_df.filter(pl.col("wall_time") == self.wall_time)
            self.df = df_step.to_pandas()
        else:
            await super().get_plant_stats()

    async def step(self, action: np.ndarray):
        reward, observation, terminal, info = await super().step(action)
        return reward, observation, terminal, info

    async def put_action(self, action: int):
        logger.debug(f"action: {action}")
        self.last_action = action

    async def sleep_until(self, wake_time: datetime):
        while (
            self.time_index + 1 < len(self.unique_wall_times)
            and self.current_time < wake_time
        ):
            self.time_index += 1
            self.wall_time = self.unique_wall_times[self.time_index]
            self.current_time = self.dataset_df.filter(
                pl.col("wall_time") == self.wall_time
            )["time"][0]

    async def close(self):
        pass
