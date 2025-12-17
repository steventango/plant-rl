# ruff: noqa
import asyncio
import io
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
from .zones import load_zone_from_config

logger = logging.getLogger("plant_rl.PlantGrowthChamber")


# Configuration for clean area calculation
# Tukey outlier detection (across plants per timestep)
TUKEY_K_UPPER = 4.0  # Conservative k factor for Tukey fence

# EWM outlier detection (within each plant over time)
CLEAN_AREA_LOWER_THRESHOLD = 0.5  # Reject if area < (1 - threshold) * ewm_mean
CLEAN_AREA_UPPER_THRESHOLD = 1.5  # Reject if area > (1 + threshold) * ewm_mean
MINIMUM_AREA_COUNT = 1  # Minimum observations before applying outlier detection
EWM_BETA = 0.1  # Decay factor for EWM (higher = smoother); alpha = 1 - beta

# Morphology features that should be replaced together with area when outlier detected
MORPHOLOGY_FEATURES = [
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


class PlantCleaningState:
    def __init__(self, features):
        self.ewm_sum = 0.0
        self.ewm_weight = 0.0
        self.prev_clean_values = {f: 0.0 for f in features}
        self.prev_clean_area = 0.0
        self.area_count = 0


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
        self.pot_quads = None
        self.dataset_path = None
        self.cv_client = CVPipelineClient()

        # Cleaning state
        self.plant_cleaning_states = []

        self.clean_areas = []
        # i.e. a mapping from datetime to the mean clean area
        self.daily_mean_clean_areas = defaultdict(float)
        self.gamma = 0.99
        self.n_step = 0
        self.duration = timedelta(minutes=1)
        self.clean_area_lower, self.clean_area_upper = 0.1, 0.3

        # Kept for backward compat / referenced in get_info but superseded by new logic
        self.uema_areas = [UEMA(alpha=0.1) for _ in range(self.zone.num_plants)]

        self.area_count = 0
        self.minimum_area_count = 5
        self.dli = 0
        self.prev_plant_areas = np.zeros(self.zone.num_plants)
        self.normalize_reward = normalize_reward

        self.last_action = np.zeros(6)
        self.last_calibrated_action = np.zeros(6)
        self.plant_areas = np.array([])
        self.last_step_time = None

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

        if self.pot_quads is None:
            if not self.is_daylight():
                logger.debug("Not daylight, skipping pot detection.")
                self.df = pd.DataFrame()
                return
            logger.debug("Daylight detected, running initial pot detection...")
            session = await self._ensure_session()
            try:
                self.pot_quads = await self.cv_client.detect_pots(session, self.image)
                if self.pot_quads:
                    num_plants = len(self.pot_quads)
                    logger.debug(f"Initialized tracking for {num_plants} plants")
                    self.uema_areas = [UEMA(alpha=0.1) for _ in range(num_plants)]
                    self.prev_plant_areas = np.zeros(num_plants)
            except Exception:
                logger.exception("Error during pot detection")
                self.pot_quads = None

        if self.pot_quads is None:
            logger.debug("No pot quads, skipping plant stats.")
            self.df = pd.DataFrame()
            return

        session = await self._ensure_session()
        iso_time = (
            self.time.isoformat(timespec="seconds").replace(":", "")
            if self.time.minute % 5 == 0
            else None
        )
        try:
            plant_stats = await self.cv_client.process_plants(
                session, self.image, self.pot_quads, timestamp_str=iso_time
            )
            self.df = pd.DataFrame(plant_stats)

            # Apply incremental cleaning
            if not self.df.empty:
                self.df = self.clean_stats_incremental(self.df)

        except Exception:
            logger.exception("Error during plant processing")
            self.df = pd.DataFrame()

    def clean_stats_incremental(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Incrementally clean plant stats using Tukey outlier detection (cross-plant)
        and EWM outlier detection (within-plant), mirroring dataset_cleaning.py.
        """
        # Ensure we have states for all plants
        num_plants = len(df)
        while len(self.plant_cleaning_states) < num_plants:
            self.plant_cleaning_states.append(PlantCleaningState(MORPHOLOGY_FEATURES))

        # 1. Cross-plant Tukey Outlier Detection
        # Compute Q1, Q3, IQR for area
        areas = df["area"].to_numpy()
        q1 = np.nanpercentile(areas, 25)
        q3 = np.nanpercentile(areas, 75)
        iqr = q3 - q1
        upper_fence = q3 + TUKEY_K_UPPER * iqr

        # Identify Tukey outliers
        tukey_outliers = (areas > upper_fence) | np.isnan(areas)

        # 2. Within-plant EWM Cleaning
        # Prepare lists for new columns
        # We need to determine available features in df that are also in MORPHOLOGY_FEATURES
        available_features = [f for f in MORPHOLOGY_FEATURES if f in df.columns]

        new_cols = defaultdict(list)
        new_cols["uema_area"] = []
        new_cols["is_outlier"] = []
        for f in available_features:
            new_cols[f"clean_{f}"] = []

        alpha = 1.0 - EWM_BETA

        for i in range(num_plants):
            state = self.plant_cleaning_states[i]
            area = areas[i]
            is_tukey_outlier = tukey_outliers[i]

            # Check for invalid area (None/NaN or <= 0 or Tukey outlier)
            is_invalid = False
            if pd.isna(area) or area <= 0 or is_tukey_outlier:
                is_invalid = True

            current_ewm = None
            if state.ewm_weight > 0:
                current_ewm = state.ewm_sum / state.ewm_weight
            else:
                current_ewm = area if not is_invalid else 0.0  # Fallback

            new_cols["uema_area"].append(current_ewm)

            # Check for temporal outlier
            is_outlier = False
            if is_invalid:
                is_outlier = True
            elif state.area_count >= MINIMUM_AREA_COUNT and state.prev_clean_area > 0:
                lower_bound = (1 - CLEAN_AREA_LOWER_THRESHOLD) * current_ewm
                upper_bound = (1 + CLEAN_AREA_UPPER_THRESHOLD) * current_ewm
                if area < lower_bound or area > upper_bound:
                    is_outlier = True

            new_cols["is_outlier"].append(is_outlier)

            if is_outlier:
                # Use previous clean values
                for f in available_features:
                    new_cols[f"clean_{f}"].append(state.prev_clean_values.get(f, 0.0))
            else:
                # Accept current values
                for f in available_features:
                    val = df.iloc[i][f]
                    new_cols[f"clean_{f}"].append(val)
                    state.prev_clean_values[f] = val

                state.prev_clean_area = area

                # Update EWM
                state.ewm_sum = state.ewm_sum * (1 - alpha) + area
                state.ewm_weight = state.ewm_weight * (1 - alpha) + 1.0
                state.area_count += 1

        # Add new columns to dataframe
        for col_name, values in new_cols.items():
            df[col_name] = values

        return df

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
        await self.sleep_until_next_step(self.duration)
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
        if self.df.empty:
            return {
                "df": self.df,
                "env_time": self.time.timestamp(),
            }
        mean = np.array(
            [self.uema_areas[i].compute() for i in range(len(self.uema_areas))]
        ).flatten()
        upper = mean * (1 + self.clean_area_upper)
        lower = mean * (1 - self.clean_area_lower)
        return {
            "df": self.df,
            "mean_clean_area": np.mean(self.clean_areas[-1]),
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

        if self.normalize_reward:
            if yesterday_morning_mean_area == 0:
                logger.debug(
                    "Yesterday's morning mean area is 0, returning 0 reward to avoid division by zero."
                )
                return 0
            reward = normalize(
                today_morning_mean_area / yesterday_morning_mean_area - 1, 0, 0.35
            )
        else:
            if len(self.clean_areas) < 10:
                logger.debug("Not enough clean areas to compute reward.")
                return 0
            reward = normalize(
                np.mean(self.clean_areas[-1]) - np.mean(self.clean_areas[-10]), 0, 150
            )

        # if reward only @ 9:30 AM
        if self.sparse_reward and not (
            self.get_local_time().hour == 9 and self.get_local_time().minute == 30
        ):
            logger.debug(
                f"Returning sparse reward of 0 at {self.get_local_time().astimezone(self.tz)}"
            )
            return 0

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
