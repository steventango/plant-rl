# ruff: noqa
import asyncio
import io
import base64
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from environments.PlantGrowthChamber.utils import create_session
from utils.constants import BALANCED_ACTION_105, DIM_ACTION
from utils.functions import normalize
from utils.metrics import UnbiasedExponentialMovingAverage as UEMA
from utils.RlGlue.environment import BaseAsyncEnvironment
import pandas as pd
from pathlib import Path
from .CVPipelineClient import CVPipelineClient
from .SmartPlugClient import POWER_KEYS, SmartPlugClient
from .zones import load_zone_from_config

logger = logging.getLogger("plant_rl.PlantGrowthChamber")


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
        self.timezone = timezone
        self.tz = ZoneInfo(timezone)
        self.sparse_reward = kwargs.get("sparse_reward", False)
        self.tz_utc = ZoneInfo("Etc/UTC")
        self.time = self.get_time()
        self.session = None
        self.pot_quads = None
        self.dataset_path = None
        self.cv_client = CVPipelineClient()

        # Smart plug
        # self.power: agent-facing carry-over of last successful reading
        # self.power_record: per-step audit value sent to CSV/WandB; sticky between fetches, None on failure
        self.smart_plug_client = (
            SmartPlugClient()
            if zone is not None and self.zone.smart_plug_host is not None
            else None
        )
        self.power = dict.fromkeys(POWER_KEYS, 0.0)
        self.power_record: dict = {}

        # Cleaning state
        self.cv_state = None
        self.last_cv_time = None
        self.cv_interval = timedelta(minutes=5)

        self.clean_areas = []
        # i.e. a mapping from datetime to the mean clean area
        self.daily_mean_clean_areas = defaultdict(float)
        self.gamma = 0.99
        self.n_step = 0
        self.duration = timedelta(minutes=1)
        self.clean_area_lower, self.clean_area_upper = 0.1, 0.3

        self.dli = 0

        self.last_action = np.zeros(6)
        self.last_calibrated_action = np.zeros(6)
        self.plant_areas = np.array([])
        self.last_step_time = None

        self.df = pd.DataFrame()

    async def _ensure_session(self):
        """Ensures an aiohttp session is available and returns it."""
        if self.session is None:
            self.session = await create_session()
        return self.session

    def set_dataset_path(self, path: Path):
        self.dataset_path = path
        self.cv_client.set_dataset_path(path)

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

        await self.get_plant_stats()
        await self.get_power()

        if not self.df.empty:
            self.plant_areas = self.df["area"].to_numpy().flatten()  # type: ignore

            # self.df["clean_area"] is already computed in get_plant_stats now
            if "clean_area" in self.df.columns:
                clean_area = self.df["clean_area"].to_numpy()
                self.clean_areas.append(clean_area)

                # Update daily mean clean areas history
                current_local_date = self.get_local_time().replace(
                    second=0, microsecond=0
                )
                mean_area_this_step = (
                    np.mean(clean_area) if clean_area.size > 0 else 0.0
                )
                self.daily_mean_clean_areas[current_local_date] = float(
                    mean_area_this_step
                )

        return self.time, self.image, self.df

    def is_daylight(self):
        local_time = self.get_local_time()
        return (
            local_time.hour > 9 or (local_time.hour == 9 and local_time.minute >= 30)
        ) and (
            local_time.hour < 20 or (local_time.hour == 20 and local_time.minute < 30)
        )

    async def get_plant_stats(self):
        assert self.image is not None, "Image must be fetched before processing."

        if not self.is_daylight():
            logger.debug("Not daylight, skipping plant stats.")
            self.df = pd.DataFrame()
            return

        # 5-minute cadence aligned with the CSV writer (env.time.minute % 5 == 0),
        # with an elapsed-time fallback so a missed boundary still triggers a fetch.
        now = self.get_time()
        on_boundary = now.minute % 5 == 0
        overdue = (
            self.last_cv_time is None or (now - self.last_cv_time) > self.cv_interval
        )
        if not (on_boundary or overdue):
            return

        session = await self._ensure_session()
        response = None
        try:
            if self.cv_state is None:
                logger.debug("Running initial CV detect...")
                response = await self.cv_client.detect(session, self.image)
            else:
                logger.debug("Running CV propagate...")
                response = await self.cv_client.propagate(
                    session, self.image, self.cv_state
                )

            if response:
                self.last_cv_time = now

                # Update state
                if "state" in response:
                    self.cv_state = response["state"]

                # Update Dataframe
                plant_stats = response.get("plant_stats", {})
                if plant_stats and not isinstance(plant_stats, list):
                    # Convert dict of dicts to list of dicts for DataFrame
                    # plant_stats is {pot_id: stats_dict, ...}
                    stats_list = []
                    for pot_id, stats in plant_stats.items():
                        if stats and "error" not in stats:
                            stats["pot_id"] = pot_id
                            stats_list.append(stats)
                    self.df = pd.DataFrame(stats_list)
                elif isinstance(plant_stats, list):
                    self.df = pd.DataFrame(plant_stats)
                else:
                    self.df = pd.DataFrame()

                # Visualization
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

    async def get_power(self):
        if self.smart_plug_client is None or self.zone.smart_plug_host is None:
            return

        reading = await self.smart_plug_client.read(self.zone.smart_plug_host)
        if reading is not None:
            self.power = reading
            self.power_record = reading
        else:
            self.power_record = dict.fromkeys(POWER_KEYS)

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
                timeout=10,
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
        self.plant_cleaning_states = []  # Reset cleaning states on start
        self.daily_mean_clean_areas = defaultdict(float)
        observation = await self.get_observation()
        self.last_step_time = self.get_time()
        self.n_step = 1
        return observation, self.get_info()

    async def step(self, action: np.ndarray):
        logger.debug(
            f"Local time: {self.get_local_time()}. Step {self.n_step} with action {action}"
        )
        if np.array_equal(action, BALANCED_ACTION_105):
            self.dli += 1.0
        elif np.array_equal(action, DIM_ACTION):
            self.dli += 0.5
        if self.get_local_time().hour == 9 and self.get_local_time().minute == 30:
            self.dli = 0.0
        await self.put_action(action)

        terminal = self.get_terminal()

        # Sleep until the next minute
        await self.sleep_until_next_step(self.duration)
        observation = await self.get_observation()
        reward = self.reward_function()
        current_time = self.get_time()
        if self.last_step_time:
            cycle_time = current_time - self.last_step_time
            warning_threshold = self.duration * 5
            if cycle_time > warning_threshold:
                logger.warning(
                    f"Cycle time ({cycle_time}) exceeded threshold ({warning_threshold})"
                )
            elif cycle_time > self.duration:
                logger.debug(
                    f"Cycle time ({cycle_time}) exceeded duration ({self.duration})"
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
        if self.df.empty:
            info = {
                "df": self.df,
                "env_time": self.time.timestamp(),
            }
        else:
            info = {
                "df": self.df,
                "mean_clean_area": np.mean(self.clean_areas[-1]),
                "env_time": self.time.timestamp(),
            }
        info.update(self.power_record)
        return info

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
            logger.debug(
                "Yesterday's morning mean area is 0, returning 0 reward to avoid division by zero."
            )
            return 0.0

        # if reward only @ 9:30 AM
        local_time = self.time.astimezone(self.tz)
        if self.sparse_reward and not (
            local_time.hour == 9 and local_time.minute == 30
        ):
            logger.debug(f"Returning sparse reward of 0 at {local_time}")
            return 0.0

        reward = np.log(today_morning_mean_area + 1) - np.log(
            yesterday_morning_mean_area + 1
        )

        if np.isnan(reward):
            logger.debug(
                f"Reward is nan: {today_morning_mean_area}, {yesterday_morning_mean_area}"
            )
            reward = 0.0

        return reward

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
