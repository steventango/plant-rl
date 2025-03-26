import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from utils.metrics import iqm
from utils.metrics import UnbiasedExponentialMovingAverage as uema
from utils.RlGlue.environment import BaseAsyncEnvironment

from .utils import process_image
from .zones import get_zone


class PlantGrowthChamber(BaseAsyncEnvironment):

    def __init__(self, zone: int, start_time: float | None = None):
        self.zone = get_zone(zone)
        self.images = {}
        self.image = None
        self.time = None
        self._start_time = start_time
        self.min_action = 0.35 * np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self.q = 0.10                     # the bottom q and the top 1-q quantiles are excluded from iqm
        self.observed_areas = []          # stores a list of arrays of observed areas in mm^2. i.e. self.observed_areas[-1] contains the latest areas of individual plants
        self.history = uema(alpha=0.01)   # history of change in average observed area over 1 time step (in units of mm^2). Note that tracing also happens at night.
        self.gamma = 1.0

    def get_raw_observation(self):
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
    
    def get_input_observation(self):
        plant_areas = self.plant_stats[:, 2].reshape(1, -1)  # TODO: Steven, this is a 1D array right?
        self.observed_areas.append(plant_areas.flatten())

        if len(self.observed_areas) >= 2:
            self.history.update(iqm(self.observed_areas[-1], self.q) - iqm(self.observed_areas[-2], self.q))
        
        time_of_day = self.transform_time_linear(self.time, total=86400/2) # time goes from 0 to 1 over 12hr period
        
        observation = np.hstack([time_of_day,
                                 self.normalize(iqm(self.observed_areas[-1], self.q)),
                                 self.normalize(self.history.compute(), l=-0.41, u=2.48)])
        
        return observation

    def step_one(self, action: np.ndarray):
        self.put_action(action)

    def put_action(self, action):
        # clip action to be between min_action and 1
        action = np.clip(action, self.min_action, 1)
        action = np.tile(action, (2, 1))
        response = self.session.put(self.zone.lightbar_url, json={"array": action.tolist()}, timeout=10)
        response.raise_for_status()

    def start(self):
        self.observed_areas = []
        self.history.reset()

        self.get_raw_observation()
        observation = self.get_input_observation()
        return observation

    def step_two(self):
        # TODO insert here a call for overnight behavior (tracing history)
        # TODO if observed_areas include overnight measurements, need to revisit reward and history definitions

        self.get_raw_observation()
        observation = self.get_input_observation()
        self.reward = self.reward_function()

        return self.reward, observation, False, self.get_info()

    def get_info(self):
        return {"gamma": self.gamma}

    def reward_function(self):
        new = self.normalize(iqm(self.observed_areas[-1], self.q))
        old = self.normalize(iqm(self.observed_areas[-2], self.q))
        return new - old

    def normalize(self, x, l=0, u=930):  # normalize area (mm^2) to between 0 and 1
        return (x - l) / (u - l)
    
    #def transform_time_sine(self, time, total=86400.0):
    #    return np.array([np.sin(2 * np.pi * time / total), np.cos(2 * np.pi * time / total)])

    def transform_time_linear(self, time, total=86400.0):
        return time / total

    def close(self):
        requests.put(self.zone.lightbar_url, json={"array": np.zeros((2, 6)).tolist()})
