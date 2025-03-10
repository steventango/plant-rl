import io
import time

import numpy as np
import requests
from PIL import Image
from datetime import datetime

from utils.RlGlue.environment import BaseAsyncEnvironment


class PlantGrowthChamber(BaseAsyncEnvironment):
    def __init__(self, camera_url: str, lightbar_url: str, start_time: float | None = None):
        self.gamma = 0.99
        self.camera_url = camera_url
        self.lightbar_url = lightbar_url
        self.image = None
        self.time = None
        self._start_time = start_time

    def get_observation(self):
        self.time = time.time() # - self.start_time
        # adjust for time zone
        self.time -= 7 * 3600
        timestamp = datetime.fromtimestamp(self.time)
        self.get_image()
        observation = (self.time, np.array(self.image))
        return observation

    def get_image(self):
        response = requests.get(self.camera_url, timeout=5)
        response.raise_for_status()
        self.image = Image.open(io.BytesIO(response.content))

    def start(self):
        # if self._start_time is None:
        #     self.start_time = time.time()
        # else:
        #     self.start_time = self._start_time
        observation = self.get_observation()
        return observation

    def step_one(self, action: np.ndarray):
        self.put_action(action)

    def step_two(self):
        observation = self.get_observation()

        # Compute reward
        self.reward = self.reward_function()

        return self.reward, observation, False, self.get_info()

    def put_action(self, action):
        action = np.tile(action, (2, 1))
        response = requests.put(self.lightbar_url, json={"array": action.tolist()}, timeout=5)
        response.raise_for_status()

    def get_info(self):
        return {"gamma": self.gamma}

    def reward_function(self):
        return np.array(self.image).mean() / 255

    def close(self):
        requests.put(self.lightbar_url, json={"array": np.zeros(6).tolist()})
