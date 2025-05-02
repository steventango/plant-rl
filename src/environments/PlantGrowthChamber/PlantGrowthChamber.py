import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from utils.RlGlue.environment import BaseAsyncEnvironment

from .cv import process_image
from .zones import get_zone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PlantGrowthChamber(BaseAsyncEnvironment):

    def __init__(self, zone: int, start_time: float | None = None, timezone: str = "Etc/UTC"):
        self.zone = get_zone(zone)
        self.images = {}
        self.image = None
        self.time = 0
        self._start_time = start_time
        # self.min_action = 0.35 * np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # TODO: save self.observed_areas at the end of the run
        self.observed_areas = (
            []
        )  # stores a list of arrays of observed areas in mm^2. i.e. self.observed_areas[-1] contains the latest areas of individual plants
        self.gamma = 0.99
        self.step = 0

        self.enforce_night = True
        self.reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])

        tz = ZoneInfo(timezone)
        dt = datetime.now(tz)
        offset = dt.utcoffset()
        assert offset is not None
        self.offset = offset.total_seconds()

    def get_observation(self):
        self.time = self.get_time()

        self.get_image()
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

    def get_image(self):

        def fetch_image(url: str):
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))

        with ThreadPoolExecutor() as executor:
            futures = {}
            if self.zone.camera_left_url:
                futures["left"] = executor.submit(fetch_image, self.zone.camera_left_url)
            if self.zone.camera_right_url:
                futures["right"] = executor.submit(fetch_image, self.zone.camera_right_url)

            for side, future in futures.items():
                self.images[side] = future.result()

    def step_one(self, action: np.ndarray):
        self.put_action(action)

    def put_action(self, action):
        # clip action to be between min_action and 1
        # action = np.clip(action, self.min_action, 1)
        # clip action to be have max value 1
        action = np.clip(action, None, 1)
        action = np.tile(action, (2, 1))
        response = self.session.put(self.zone.lightbar_url, json={"array": action.tolist()}, timeout=10)
        response.raise_for_status()

    def start(self):
        # TODO: deal with start logic...
        self.observed_areas = []
        observation = self.get_observation()
        self.step += 1
        return observation, self.get_info()

    def step_two(self):
        terminal = False
        logger.info(f"Step {self.step} at time {self.time}")
        if self.enforce_night:
            local_time = self.time + self.offset
            logger.info(f"Local time: {local_time}")
            clock_time = local_time % 86400
            logger.info(f"Clock time: {int(clock_time // 3600)}:{int((clock_time % 3600) // 60)}:{clock_time % 60}")
            night_start = 21 * 3600 + 00 * 60
            night_end = 9 * 3600 + 00 * 60
            if night_end < night_start:
                is_night = clock_time < night_end or clock_time >= night_start
                seconds_to_wait = 86400 - (clock_time) + night_end
            else:
                is_night = night_start <= clock_time < night_end
                seconds_to_wait = night_end - clock_time
            if is_night:
                terminal = True
                self.put_action(np.zeros(6))
                logger.info(f"Nighttime enforced. Waiting for {seconds_to_wait} seconds.")
                time.sleep(seconds_to_wait)
                self.put_action(self.reference_spectrum)
                logger.info("Nighttime ended. Reference spectrum applied.")
                time.sleep(600)

        observation = self.get_observation()
        self.reward = self.reward_function()
        self.step += 1
        logger.info(f"Step {self.step} completed. Reward: {self.reward}, Terminal: {terminal}")

        return self.reward, observation, terminal, self.get_info()

    def get_info(self):
        return {"df": self.df}

    def reward_function(self):
        new = np.mean(self.observed_areas[-1])
        old = np.mean(self.observed_areas[-2])
        return new - old

    def close(self):
        self.put_action(np.zeros(6))
