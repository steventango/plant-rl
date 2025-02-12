import io
import time

import numpy as np
import requests
from PIL import Image
from RlGlue.environment import BaseEnvironment


class PlantGrowthChamber(BaseEnvironment):
    def __init__(self, camera_url: str, lightbar_url: str):
        self.gamma = 0.99
        self.camera_url = camera_url
        self.lightbar_url = lightbar_url

    def get_observation(self):
        response = requests.get(self.camera_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        array = np.array(image)
        return array

    def start(self):
        self.time = time.time()
        self.current_state = self.get_observation()
        return self.current_state

    def step(self, action: np.ndarray):
        response = requests.put(self.lightbar_url, json={"array": action.tolist()})
        response.raise_for_status()

        # Define state
        self.current_state = self.get_observation()

        # Compute reward
        self.reward = self.reward_function()

        return self.reward, self.current_state, False, self.get_info()

    def get_info(self):
        return {"gamma": self.gamma}

    def reward_function(self):
        return 0

    def close(self):
        requests.put(self.lightbar_url, json={"array": np.zeros(6).tolist()})
