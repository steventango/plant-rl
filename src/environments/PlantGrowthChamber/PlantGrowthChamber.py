import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from utils.metrics import UnbiasedExponentialMovingAverage
from utils.RlGlue.environment import BaseAsyncEnvironment

from .utils import process_image
from .zones import get_zone


class PlantGrowthChamber(BaseAsyncEnvironment):

    def __init__(self, zone: int, start_time: float | None = None):
        self.gamma = 0.99
        self.zone = get_zone(zone)
        self.images = {}
        self.image = None
        self.time = None
        self.lag = 1
        self._start_time = start_time
        # todo don't hardcode 16
        self.plant_area_emas = [UnbiasedExponentialMovingAverage(alpha=0.1)] * 16
        self.min_action = 0.35 * np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_observation(self):
        self.time = datetime.now().timestamp()

        self.get_image()
        if "left" in self.images and "right" in self.images:
            self.image = np.hstack((np.array(self.images["left"]), np.array(self.images["right"])))
        elif "left" in self.images:
            self.image = np.array(self.images["left"])
        elif "right" in self.images:
            self.image = np.array(self.images["right"])

        self.df = process_image(self.image, self.zone.trays, self.images)

        self.plant_stats = np.array(self.df, dtype=np.float32)

        plant_area_emas_prev = [plant_area_ema.compute() for plant_area_ema in self.plant_area_emas]
        self.mean_plant_area_ema_prev = np.mean(plant_area_emas_prev)

        plant_areas = self.plant_stats[:, 2]
        for plant_area_ema in self.plant_area_emas:
            plant_area_ema.update(values=plant_areas)

        plant_area_emas = [plant_area_ema.compute() for plant_area_ema in self.plant_area_emas]
        self.mean_plant_area_ema = np.mean(plant_area_emas)

        return self.time, self.image, self.plant_stats

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

    def start(self):
        observation = self.get_observation()
        return observation

    def step_one(self, action: np.ndarray):
        self.put_action(action)

    def step_two(self):
        observation = self.get_observation()
        self.reward = self.reward_function()
        return self.reward, observation, False, self.get_info()

    def put_action(self, action):
        # clip action to be between min_action and 1
        action = np.clip(action, self.min_action, 1)
        action = np.tile(action, (2, 1))
        response = self.session.put(self.zone.lightbar_url, json={"array": action.tolist()}, timeout=10)
        response.raise_for_status()

    def get_info(self):
        return {"gamma": self.gamma}

    def reward_function(self):
        new = self.normalize(self.mean_plant_area_ema)
        old = self.normalize(self.mean_plant_area_ema_prev)
        return (new / old - 1).item()

    def normalize(self, x):  # normalize observation to between 0 and 1
        # TODO: check if this number is too big?
        u = 30000  # max historic area of one plant (in pixels)
        l = 0
        return (x - l) / (u - l)

    def close(self):
        requests.put(self.zone.lightbar_url, json={"array": np.zeros((2, 6)).tolist()})
