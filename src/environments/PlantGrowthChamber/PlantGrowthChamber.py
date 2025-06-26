import asyncio
import io
import logging
from collections import defaultdict  # Added import
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from environments.PlantGrowthChamber.utils import create_session  # Changed import
from utils.functions import normalize
from utils.metrics import UnbiasedExponentialMovingAverage as UEMA
from utils.RlGlue.environment import BaseAsyncEnvironment

from .cv import process_image
from .zones import load_zone_from_config

logger = logging.getLogger("PlantGrowthChamber")
logger.setLevel(logging.DEBUG)


class PlantGrowthChamber(BaseAsyncEnvironment):
    def __init__(
        self,
        zone: str | None = None,
        timezone: str = "Etc/UTC",
        normalize_reward: bool = False,
        **kwargs,
    ):
        if zone is not None:
            self.zone = load_zone_from_config(zone) if isinstance(zone, str) else zone
        self.images = {}
        self.image = None
        self.tz = ZoneInfo(timezone)
        self.sparse_reward = kwargs.get("sparse_reward", False)
        self.tz_utc = ZoneInfo("Etc/UTC")
        self.time = self.get_time()
        self.session = None

        self.clean_areas = []
        # i.e. a mapping from datetime to the mean clean area
        self.daily_mean_clean_areas = defaultdict(float)
        self.gamma = 0.99
        self.n_step = 0
        self.duration = timedelta(minutes=1)
        self.clean_area_lower, self.clean_area_upper = 0.1, 0.3
        self.uema_areas = [UEMA(alpha=0.1) for _ in range(self.zone.num_plants)]
        self.area_count = 0
        self.minimum_area_count = 5
        self.prev_plant_areas = np.zeros(self.zone.num_plants)
        self.normalize_reward = normalize_reward
        self.glue = None

        self.last_action = None
        self.plant_areas = np.array([])

    async def _ensure_session(self):
        """Ensures an aiohttp session is available and returns it."""
        if self.session is None:
            self.session = await create_session()
        return self.session

    async def get_observation(self):
        self.time = self.get_time()

        await self.get_image()
        if "left" in self.images and "right" in self.images:
            self.image = np.hstack(
                (np.array(self.images["left"]), np.array(self.images["right"]))
            )
        elif "left" in self.images:
            self.image = np.array(self.images["left"])
        elif "right" in self.images:
            self.image = np.array(self.images["right"])

        self.get_plant_stats()

        if not self.df.empty:
            self.plant_areas = self.df["area"].to_numpy().flatten()  # type: ignore

            clean_area = self.get_clean_area(self.plant_areas)
            self.df["clean_area"] = clean_area

            self.clean_areas.append(clean_area)

            # Update daily mean clean areas history
            current_local_date = self.get_local_time().replace(second=0, microsecond=0)
            mean_area_this_step = np.mean(clean_area) if clean_area.size > 0 else 0.0
            self.daily_mean_clean_areas[current_local_date] = float(mean_area_this_step)

        return self.time, self.image, self.df

    def get_plant_stats(self):
        assert self.image is not None, "Image must be fetched before processing."
        self.df, self.detections = process_image(
            self.image, self.zone.trays, self.images
        )
        self.plant_stats = np.array(self.df, dtype=np.float32)

    def get_clean_area(self, plant_areas):
        clean_area = plant_areas.copy()
        mean = np.array(
            [self.uema_areas[i].compute() for i in range(self.zone.num_plants)]
        ).flatten()
        cond = (self.area_count > self.minimum_area_count) & (
            (plant_areas < (1 - self.clean_area_lower) * mean)
            | (plant_areas > (1 + self.clean_area_upper) * mean)
        )
        clean_area[cond] = self.prev_plant_areas[cond]
        self.prev_plant_areas[~cond] = plant_areas[~cond]
        for i, area in enumerate(plant_areas):
            if area > 0:
                self.uema_areas[i].update(area)
        self.area_count += 1
        return clean_area

    def get_time(self):
        return datetime.now(tz=self.tz_utc)

    def get_local_time(self):
        return self.get_time().astimezone(self.tz)

    async def get_image(self):
        """Fetch images from cameras using aiohttp"""
        tasks = []
        session = await self._ensure_session()
        if self.zone.camera_left_url:
            tasks.append(self._fetch_image(session, self.zone.camera_left_url, "left"))
        if self.zone.camera_right_url:
            tasks.append(
                self._fetch_image(session, self.zone.camera_right_url, "right")
            )

        if tasks:
            await asyncio.gather(*tasks)

    async def _fetch_image(self, session, url, side):
        """Helper method to fetch a single image with retry logic"""
        try:
            async with session.get(url, timeout=60) as response:
                image_data = await response.read()
                self.images[side] = Image.open(io.BytesIO(image_data))
                logger.debug(f"Successfully fetched image from {url}")
        except Exception as e:
            logger.error(f"Error fetching image from {url} after retries: {str(e)}")
            # Keep previous image if available, otherwise this side will be missing

    async def put_action(self, action):
        """Send action to the lightbar using aiohttp with retry logic"""
        self.last_action = action

        # clip action to have max value 1
        action = np.clip(action, None, 1)
        action = np.tile(action, (2, 1))

        try:
            session = await self._ensure_session()
            logger.debug(f"{self.zone.lightbar_url}: {action}")
            assert self.zone.lightbar_url is not None, "Lightbar URL must be set."
            await session.put(
                self.zone.lightbar_url, json={"array": action.tolist()}, timeout=10
            )
        except Exception as e:
            logger.error(f"Error: {self.zone.lightbar_url} after retries: {str(e)}")
            raise

    async def start(self):
        logger.info(f"Local time: {self.get_local_time()}. Step 0")
        self.n_step = 0
        self.clean_areas = []
        self.daily_mean_clean_areas = defaultdict(float)
        await self.sleep_until_next_step(self.duration)
        observation = await self.get_observation()
        self.n_step = 1
        return observation, self.get_info()

    async def step(self, action: np.ndarray):
        logger.info(
            f"Local time: {self.get_local_time()}. Step {self.n_step} with action {action}"
        )
        await self.put_action(action)

        terminal = self.get_terminal()

        # Sleep until the next minute
        await self.sleep_until_next_step(self.duration)
        observation = await self.get_observation()
        reward = self.reward_function()
        logger.info(
            f"Local time: {self.get_local_time()}. Step {self.n_step} completed. Reward: {reward}, Terminal: {terminal}"
        )
        self.n_step += 1

        return reward, observation, terminal, self.get_info()

    def get_next_step_time(self, duration: timedelta):
        duration_s = duration.total_seconds()
        wake_time = datetime.fromtimestamp(
            (self.get_time().timestamp() // duration_s + 1) * duration_s, tz=self.tz_utc
        )
        return wake_time

    async def sleep_until(self, wake_time: datetime):
        time_left = wake_time - self.get_time()
        logger.info(f"Sleeping until {wake_time.astimezone(self.tz)} (in {time_left})")
        await asyncio.sleep(time_left.total_seconds())

    async def sleep_until_next_step(self, duration: timedelta):
        next_step_time = self.get_next_step_time(duration)
        await self.sleep_until(next_step_time)

    def get_info(self):
        if self.df.empty:
            return {
                "df": self.df,
                "env_time": self.time.timestamp(),
            }
        N = 3
        raw_area = self.plant_areas[:N]
        mean = np.array(
            [self.uema_areas[i].compute() for i in range(self.zone.num_plants)]
        ).flatten()[:N]
        upper = mean * (1 + self.clean_area_upper)
        lower = mean * (1 - self.clean_area_lower)
        return {
            "df": self.df,
            "mean_clean_area": np.mean(self.clean_areas[-1]),
            "clean_area": self.clean_areas[-1][:N],
            "raw_area": raw_area,
            "uema_area": mean,
            "upper_area": upper,
            "lower_area": lower,
            "env_time": self.time.timestamp(),
        }

    def get_terminal(self) -> bool:
        return False

    def reward_function(self):
        today_morning_local_date = self.get_local_time().replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        yesterday_morning_local_date = today_morning_local_date - timedelta(days=1)

        today_morning_mean_area = self.daily_mean_clean_areas.get(
            today_morning_local_date, 0.0
        )
        yesterday_morning_mean_area = self.daily_mean_clean_areas.get(
            yesterday_morning_local_date, 0.0
        )

        if yesterday_morning_mean_area == 0:
            return 0

        if self.normalize_reward:
            reward = normalize(
                today_morning_mean_area / yesterday_morning_mean_area - 1, 0, 0.35
            )
        else:
            reward = normalize(
                today_morning_mean_area - yesterday_morning_mean_area, 0, 50
            )

        # if reward only @ 9:30 AM
        if self.sparse_reward and not (
            self.get_local_time().hour == 9 and self.get_local_time().minute == 30
        ):
            return 0

        return reward

    async def close(self):
        """Close the environment and clean up resources."""
        # Turn off lights
        try:
            await self.put_action(np.zeros(6))
            logger.info(
                f"Lights turned off for zone {self.zone.identifier} during environment closure"
            )
        except Exception as e:
            logger.error(
                f"Failed to turn off lights for zone {self.zone.identifier} during close: {e}"
            )

        # Close the aiohttp RetryClient
        if self.session:
            try:
                await self.session.close()
                logger.info(f"Closed aiohttp session for zone {self.zone.identifier}")
            except Exception as e:
                logger.error(
                    f"Error closing aiohttp session for zone {self.zone.identifier}: {str(e)}"
                )
            self.session = None
