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

    def __init__(self, zone: int, start_time: float | None = None, timezone: str = "Etc/UTC"):
        self.zone = get_zone(zone)
        self.images = {}
        self.image = None
        self.time = 0

        self.observed_areas = []
        # stores a list of arrays of observed areas in mm^2.
        # i.e. self.observed_areas[-1] contains the latest areas of individual plants
        self.gamma = 0.99
        self.n_step = 0
        self.duration = 60

        self.enforce_night = True
        self.reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])

        self.tz = ZoneInfo(timezone)
        dt = datetime.now(self.tz)
        offset = dt.utcoffset()
        assert offset is not None
        self.offset = offset.total_seconds()
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
        return datetime.now().timestamp()

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
        await self.put_action(self.reference_spectrum)
        # TODO: deal with start logic...
        self.observed_areas = []
        # calculate the time left until the next round duration
        next_time = datetime.fromtimestamp((datetime.now().timestamp() // self.duration + 1) * self.duration)
        logger.info(f"Next round time: {next_time}")
        time_left = next_time - datetime.now()
        logger.info(f"Time left until start: {time_left}")
        await asyncio.sleep(time_left.total_seconds())
        observation = await self.get_observation()
        self.n_step += 1
        return observation, self.get_info()

    async def step(self, action: np.ndarray):
        self.last_action = action
        logger.info(f"Step {self.n_step} with action {action}")
        await self.put_action(action)
        terminal = False

        if self.enforce_night:
            if self.is_night():
                time_to_wait = self.get_time_until_night_end()
                terminal = True
                await self.put_action(np.zeros(6))
                logger.info(f"Nighttime enforced. Waiting for {time_to_wait}.")
                await asyncio.sleep(time_to_wait.total_seconds())
                await self.put_action(self.reference_spectrum)
                logger.info("Nighttime ended. Reference spectrum applied.")

        # calculate the time left until the next step
        next_time = datetime.fromtimestamp((datetime.now().timestamp() // self.duration + 1) * self.duration)
        logger.info(f"Next time: {next_time}")
        time_left = next_time - datetime.now()
        logger.info(f"Time left until next time: {time_left}")
        await asyncio.sleep(time_left.total_seconds())
        observation = await self.get_observation()
        self.reward = self.reward_function()
        logger.info(f"Step {self.n_step} completed. Reward: {self.reward}, Terminal: {terminal}")
        self.n_step += 1

        return self.reward, observation, terminal, self.get_info()

    def get_info(self):
        return {"df": self.df}

    def reward_function(self):
        new = np.mean(self.observed_areas[-1])
        old = np.mean(self.observed_areas[-2])
        return new - old

    # def is_night(self):
    #     local_time = datetime.now(tz=self.tz)
    #     logger.info(f"Local time: {local_time}")
    #     night_start = datetime.now(tz=self.tz).replace(hour=21, minute=0, second=0, microsecond=0)
    #     night_end = datetime.now(tz=self.tz).replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    #     is_night = night_start <= local_time < night_end
    #     return is_night

    def is_night(self):
        local_time = datetime.now(tz=self.tz)
        logger.info(f"Local time: {local_time}")
        is_night = local_time.hour % 2 == 0
        return is_night

    def get_time_until_night_end(self):
        local_time = datetime.now(tz=self.tz)
        night_end = local_time + timedelta(hours=1)
        night_end = night_end.replace(minute=0, second=0, microsecond=0)
        return night_end - local_time

    # def get_time_until_night_end(self):
    #     local_time = datetime.now(tz=self.tz)
    #     night_end = datetime.now(tz=self.tz).replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    #     return night_end - local_time

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
