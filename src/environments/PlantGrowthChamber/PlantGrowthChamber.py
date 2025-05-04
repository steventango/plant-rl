import asyncio
import io
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np
from aiohttp_retry import ExponentialRetry, RetryClient
from PIL import Image

from utils.RlGlue.environment import BaseAsyncEnvironment

from .cv import process_image
from .zones import get_zone

logger = logging.getLogger("PlantGrowthChamber")
logger.setLevel(logging.DEBUG)

_session = None


async def get_session():
    global _session
    if _session is not None:
        return _session
    # Configure retry options with exponential backoff
    retry_options = ExponentialRetry(
        attempts=3,  # Maximum 3 retry attempts
        start_timeout=0.5,  # Start with 0.5s delay
        max_timeout=10,  # Maximum 10s delay
        factor=2,  # Double the delay each retry
        statuses={500, 502, 503, 504, 429},  # Retry on server errors and rate limiting
    )

    # Create RetryClient with retry options
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    _session = RetryClient(
        client_session=aiohttp.ClientSession(timeout=timeout, connector=connector),
        retry_options=retry_options,
        raise_for_status=True,  # Automatically raise for HTTP errors
    )
    return _session


class PlantGrowthChamber(BaseAsyncEnvironment):

    def __init__(self, zone: int, timezone: str = "Etc/UTC"):
        self.zone = get_zone(zone)
        self.images = {}
        self.image = None
        self.tz = ZoneInfo(timezone)
        self.tz_utc = ZoneInfo("Etc/UTC")
        self.time = self.get_time()

        self.observed_areas = []
        # stores a list of arrays of observed areas in mm^2.
        # i.e. self.observed_areas[-1] contains the latest areas of individual plants
        self.gamma = 0.99
        self.n_step = 0
        self.duration = timedelta(seconds=60)

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

        self.df, self.detections = process_image(self.image, self.zone.trays, self.images)

        self.plant_stats = np.array(self.df, dtype=np.float32)

        plant_areas = self.plant_stats[:, 2].reshape(1, -1)
        self.observed_areas.append(plant_areas.flatten())

        return self.time, self.image, self.plant_stats

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
        if self.enforce_night and self.is_night():
            await self.lights_off_and_sleep_until_morning()
        await self.put_action(self.dim_action)
        self.observed_areas = []
        await self.sleep_until_next_step(self.duration)
        observation = await self.get_observation()
        self.n_step += 1
        return observation, self.get_info()

    async def step(self, action: np.ndarray):
        logger.info(f"Step {self.n_step} with action {action}")
        terminal = False

        duration = self.duration
        if self.enforce_night and self.is_night():
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
        is_night = 20 <= local_time.minute < 22
        logger.info(f"Local time: {local_time}, is_night: {is_night}")
        return is_night

    def get_next_step_time(self, duration: timedelta):
        duration_s = duration.total_seconds()
        wake_time = datetime.fromtimestamp((self.get_time().timestamp() // duration_s + 1) * duration_s, tz=self.tz_utc)
        return wake_time

    def get_morning_time(self):
        local_time = self.get_local_time()
        morning_time = local_time.replace(hour=18, minute=22, second=0, microsecond=0)
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
        return {"df": self.df}

    def reward_function(self):
        new = np.mean(self.observed_areas[-1])
        old = np.mean(self.observed_areas[-2])
        return new - old

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
