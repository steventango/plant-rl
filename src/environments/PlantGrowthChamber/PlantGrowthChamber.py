import asyncio
import base64
import io
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image

from environments.PlantGrowthChamber.utils import create_session
from utils.RlGlue.environment import BaseAsyncEnvironment

from .CVPipelineClient import CVPipelineClient
from .zones import load_zone_from_config

logger = logging.getLogger("plant-data.PlantGrowthChamber")


class PlantGrowthChamber(BaseAsyncEnvironment):
    def __init__(
        self,
        zone: str | None = None,
        timezone: str = "Etc/UTC",
        **kwargs,
    ):
        if zone is not None:
            self.zone = load_zone_from_config(zone) if isinstance(zone, str) else zone
        self.images = {}
        self.image = None
        self.tz = ZoneInfo(timezone)
        self.tz_utc = ZoneInfo("Etc/UTC")
        self.timezone = timezone
        self.time = self.get_time()
        self.session = None

        self.cv_client = CVPipelineClient()
        self.cv_state = None
        self.df = pd.DataFrame()

        self.image_interval = timedelta(minutes=10)
        self.last_image_time: datetime | None = None
        self.images_captured: bool = False

        self.clean_areas = []
        self.daily_mean_clean_areas = defaultdict(float)
        self.n_step = 0
        self.duration = timedelta(minutes=1)

        self.last_action = np.zeros(6)
        self.last_calibrated_action = np.zeros(6)
        self.last_step_time = None

    async def _ensure_session(self):
        """Ensures an aiohttp session is available and returns it."""
        if self.session is None:
            self.session = await create_session()
        return self.session

    async def get_observation(self):
        self.time = self.get_time()

        current_bucket = self.time.replace(
            minute=(self.time.minute // 10) * 10, second=0, microsecond=0
        )
        should_capture = (
            self.last_image_time is None or current_bucket > self.last_image_time
        )
        self.images_captured = should_capture

        if should_capture:
            self.last_image_time = current_bucket
            await self.get_image()

            if "left" in self.images and "right" in self.images:
                self.image = np.hstack(
                    (np.array(self.images["left"]), np.array(self.images["right"]))
                )
            elif "left" in self.images:
                self.image = np.array(self.images["left"])
            elif "right" in self.images:
                self.image = np.array(self.images["right"])

            await self.get_plant_stats()

            if not self.df.empty and "clean_area" in self.df.columns:
                clean_area = self.df["clean_area"].to_numpy()
                self.clean_areas.append(clean_area)
                current_local_date = self.get_local_time().replace(
                    second=0, microsecond=0
                )
                self.daily_mean_clean_areas[current_local_date] = float(
                    np.mean(clean_area) if clean_area.size > 0 else 0.0
                )

        return self.time, self.image, self.df

    def is_daylight(self):
        local_time = self.get_local_time()
        return 9 <= local_time.hour < 21

    async def get_plant_stats(self):
        assert self.image is not None, "Image must be fetched before processing."

        if not self.is_daylight():
            self.df = pd.DataFrame()
            return

        session = await self._ensure_session()
        try:
            if self.cv_state is None:
                response = await self.cv_client.detect(session, self.image)
            else:
                response = await self.cv_client.propagate(
                    session, self.image, self.cv_state
                )

            if response:
                if "state" in response:
                    self.cv_state = response["state"]

                plant_stats = response.get("plant_stats", {})
                if plant_stats and not isinstance(plant_stats, list):
                    stats_list = [
                        {**stats, "pot_id": pot_id}
                        for pot_id, stats in plant_stats.items()
                        if stats and "error" not in stats
                    ]
                    self.df = pd.DataFrame(stats_list)
                elif isinstance(plant_stats, list):
                    self.df = pd.DataFrame(plant_stats)
                else:
                    self.df = pd.DataFrame()

                if "visualization_data" in response and response["visualization_data"]:
                    try:
                        vis_data = base64.b64decode(response["visualization_data"])
                        self.images["visualization"] = Image.open(io.BytesIO(vis_data))
                    except Exception:
                        logger.warning(
                            "Failed to decode visualization image", exc_info=True
                        )
        except Exception:
            logger.exception("Error during CV processing")
            self.df = pd.DataFrame()

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
        except Exception:
            logger.warning(
                f"Warning: {url} after retries, re-using previous image", exc_info=True
            )
            # Keep previous image if available, otherwise this side will be missing

    async def put_action(self, action):
        """Send action to the lightbar using aiohttp with retry logic"""
        last_calibrated_action = (
            self.zone.calibration.get_calibrated_action(action)
            if self.zone.calibration
            else action
        )

        # clip action to have max value 1
        last_calibrated_action = np.clip(last_calibrated_action, None, 1)
        action_to_send = np.tile(last_calibrated_action, (2, 1))

        try:
            session = await self._ensure_session()
            logger.debug(f"{self.zone.lightbar_url}: {action_to_send}")
            assert self.zone.lightbar_url is not None, "Lightbar URL must be set."
            await session.put(
                self.zone.lightbar_url,
                json={"array": action_to_send.tolist()},
                timeout=2,
            )
            self.last_action = action
            self.last_calibrated_action = last_calibrated_action
        except Exception:
            if not np.array_equal(action, self.last_action):
                logger.exception(
                    f"Error: {self.zone.lightbar_url} after retries, re-using last action: {self.last_action}"
                )
            else:
                logger.warning(
                    f"Warning: {self.zone.lightbar_url} after retries, last action was identical",
                    exc_info=True,
                )

    async def start(self):
        logger.debug(f"Local time: {self.get_local_time()}. Step 0")
        self.n_step = 0
        self.clean_areas = []
        self.daily_mean_clean_areas = defaultdict(float)
        observation = await self.get_observation()
        await self.sleep_until_next_step(self.duration)
        self.last_step_time = self.get_time()
        self.n_step = 1
        return observation, self.get_info()

    async def step(self, action: np.ndarray):
        logger.debug(
            f"Local time: {self.get_local_time()}. Step {self.n_step} with action {action}"
        )
        await self.put_action(action)

        terminal = self.get_terminal()

        # Sleep until the next minute
        await self.sleep_until_next_step(self.duration)
        observation = await self.get_observation()
        reward = self.reward_function()
        current_time = self.get_time()
        if self.last_step_time:
            cycle_time = current_time - self.last_step_time
            warning_threshold = self.duration * 1.5
            if cycle_time > warning_threshold:
                logger.warning(
                    f"Cycle time ({cycle_time}) exceeded duration by 50% ({warning_threshold})"
                )
            elif cycle_time > self.duration:
                logger.debug(
                    f"Cycle time ({cycle_time}) exceeded duration {self.duration})"
                )
        self.last_step_time = current_time
        logger.debug(
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
        logger.debug(f"Sleeping until {wake_time.astimezone(self.tz)} (in {time_left})")
        await asyncio.sleep(time_left.total_seconds())

    async def sleep_until_next_step(self, duration: timedelta):
        next_step_time = self.get_next_step_time(duration)
        await self.sleep_until(next_step_time)

    def get_info(self):
        return {}

    def get_terminal(self) -> bool:
        return False

    def reward_function(self):
        local_now = self.get_local_time()
        if local_now.hour != 9 or local_now.minute != 0:
            return 0

        today_morning_local_date = local_now.replace(second=0, microsecond=0)
        yesterday_morning_local_date = today_morning_local_date - timedelta(days=1)

        today_morning_mean_area = self.daily_mean_clean_areas.get(
            today_morning_local_date, 0.0
        )
        yesterday_morning_mean_area = self.daily_mean_clean_areas.get(
            yesterday_morning_local_date, 0.0
        )

        if yesterday_morning_mean_area == 0:
            logger.debug(
                "Yesterday's morning mean area is 0, returning 0 reward to avoid division by zero."
            )
            return 0
        else:
            return today_morning_mean_area / yesterday_morning_mean_area - 1

    async def close(self):
        """Close the environment and clean up resources."""
        # Close the aiohttp RetryClient
        if self.session:
            try:
                await self.session.close()
                logger.debug(f"Closed aiohttp session for zone {self.zone.identifier}")
            except Exception as e:
                logger.exception(
                    f"Error closing aiohttp session for zone {self.zone.identifier}: {str(e)}"
                )
            self.session = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if "session" in state:
            del state["session"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.session = None
