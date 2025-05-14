import asyncio
import io
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from environments.PlantGrowthChamber.utils import get_session
from utils.RlGlue.environment import BaseAsyncEnvironment
from utils.metrics import UnbiasedExponentialMovingAverage as UEMA
from .cv import process_image
from .zones import get_zone

logger = logging.getLogger("PlantGrowthChamber")
logger.setLevel(logging.DEBUG)


def normalize(x, lower, upper):
    return (x - lower) / (upper - lower)


class PlantGrowthChamber(BaseAsyncEnvironment):

    def __init__(self, zone: int | None = None, timezone: str = "Etc/UTC", **kwargs):
        if zone is not None:
            self.zone = get_zone(zone) if isinstance(zone, int) else zone
        self.images = {}
        self.image = None
        self.tz = ZoneInfo(timezone)
        self.tz_utc = ZoneInfo("Etc/UTC")
        self.time = self.get_time()

        self.clean_areas = []
        # stores a list of arrays of observed areas in mm^2.
        # i.e. self.clean_areas[-1] contains the latest areas of individual plants
        self.gamma = 0.99
        self.n_step = 0
        self.duration = timedelta(minutes=10)
        self.clean_area_lower, self.clean_area_upper = 0.1, 0.3
        self.uema_areas = [UEMA(alpha=0.1) for _ in range(self.zone.num_plants)]
        self.area_count = 0
        self.minimum_area_count = 5
        self.prev_plant_areas = np.zeros(self.zone.num_plants)
        self.enforce_night = True
        self.dim_action = 0.675 * np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])

        self.last_action = None

    async def get_observation(self):
        self.time = self.get_time()

        await self.get_image()
        if "left" in self.images and "right" in self.images:
            self.image = np.hstack((np.array(self.images["left"]), np.array(self.images["right"])))
        elif "left" in self.images:
            self.image = np.array(self.images["left"])
        elif "right" in self.images:
            self.image = np.array(self.images["right"])

        self.get_plant_stats()

        self.plant_areas = self.df["area"].to_numpy().flatten()

        clean_area = self.get_clean_area(self.plant_areas)

        self.clean_areas.append(clean_area)

        return self.time, self.image, self.plant_stats

    def get_plant_stats(self):
        self.df, self.detections = process_image(self.image, self.zone.trays, self.images)
        self.plant_stats = np.array(self.df, dtype=np.float32)

    def get_clean_area(self, plant_areas):
        clean_area = plant_areas.copy()
        mean = np.array([self.uema_areas[i].compute() for i in range(self.zone.num_plants)]).flatten()
        cond = (self.area_count > self.minimum_area_count) & ((plant_areas < (1 - self.clean_area_lower) * mean) | (
            plant_areas > (1 + self.clean_area_upper) * mean
        ))
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
        session = await get_session()
        if self.zone.camera_left_url:
            tasks.append(self._fetch_image(session, self.zone.camera_left_url, "left"))
        if self.zone.camera_right_url:
            tasks.append(self._fetch_image(session, self.zone.camera_right_url, "right"))

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
            session = await get_session()
            logger.debug(f"{self.zone.lightbar_url}: {action}")
            await session.put(self.zone.lightbar_url, json={"array": action.tolist()}, timeout=10)
        except Exception as e:
            logger.error(f"Error: {self.zone.lightbar_url} after retries: {str(e)}")
            raise

    async def start(self):
        self.n_step = 0
        if self.enforce_night and self.is_night():
            await self.lights_off_and_sleep_until_morning()
        await self.put_action(self.dim_action)
        self.clean_areas = []
        await self.sleep_until_next_step(self.duration)
        observation = await self.get_observation()
        self.n_step = 1
        return observation, self.get_info()

    async def step(self, action: np.ndarray):
        logger.info(f"Step {self.n_step} with action {action}")
        terminal = self.get_terminal()

        duration = self.duration
        if self.enforce_night and self.is_night() and not terminal:
            terminal = True
            await self.lights_off_and_sleep_until_morning()
            action = self.dim_action
            logger.info("Nighttime ended. Reference spectrum applied.")
            duration /= 2

        await self.put_action(action)

        # calculate the time left until the next step
        await self.sleep_until_next_step(duration)
        observation = await self.get_observation()
        self.reward = self.reward_function()
        logger.info(f"Step {self.n_step} completed. Reward: {self.reward}, Terminal: {terminal}")
        self.n_step += 1

        return self.reward, observation, terminal, self.get_info()

    def is_night(self):
        local_time = self.get_local_time()
        is_night = local_time.hour >= 21 or local_time.hour < 9
        logger.info(f"Local time: {local_time}, is_night: {is_night}")
        return is_night

    def get_next_step_time(self, duration: timedelta):
        duration_s = duration.total_seconds()
        wake_time = datetime.fromtimestamp((self.get_time().timestamp() // duration_s + 1) * duration_s, tz=self.tz_utc)
        return wake_time

    def get_morning_time(self):
        local_time = self.get_local_time()
        local_time_greater_than_9 = local_time.hour >= 9
        morning_time = local_time.replace(hour=9, minute=0, second=0, microsecond=0)
        if local_time_greater_than_9:
            morning_time += timedelta(days=1)
        return morning_time

    async def sleep_until(self, wake_time: datetime):
        time_left = wake_time - self.get_time()
        logger.info(f"Sleeping until {wake_time} (in {time_left})")
        await asyncio.sleep(time_left.total_seconds())

    async def sleep_until_next_step(self, duration: timedelta):
        next_step_time = self.get_next_step_time(duration)
        await self.sleep_until(next_step_time)

    async def lights_off_and_sleep_until_morning(self):
        action = np.zeros(6)
        await self.put_action(action)
        logger.info(f"Nighttime enforced!")
        morning_time = self.get_morning_time()
        await self.sleep_until(morning_time)

    def get_info(self):
        N = 3
        raw_area = self.plant_areas[:N]
        mean = np.array([self.uema_areas[i].compute() for i in range(self.zone.num_plants)]).flatten()[:N]
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
        }

    def get_terminal(self) -> bool:
        return False

    def reward_function(self):
        new = np.mean(self.clean_areas[-1])
        old = np.mean(self.clean_areas[-2])
        if old == 0:
            return 0
        reward = new - old
        normalized_reward = normalize(reward, 0, 50)
        return normalized_reward

    async def close(self):
        """Close the environment and clean up resources."""
        # Turn off lights
        try:
            await self.put_action(np.zeros(6))
            logger.info("Lights turned off during environment closure")
        except Exception as e:
            logger.error(f"Failed to turn off lights during close: {e}")

        # Close the aiohttp RetryClient
        try:
            session = await get_session()
            await session.close()
            logger.info("Closed aiohttp RetryClient")
        except Exception as e:
            logger.error(f"Error closing aiohttp session: {str(e)}")
