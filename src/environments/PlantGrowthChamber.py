import io
import time

import numpy as np
import requests
from PIL import Image
from RlGlue.environment import BaseEnvironment


class PlantGrowthChamber(BaseEnvironment):
    def __init__(self, camera_url_0: str, camera_url_1: str, lightbar_url: str):
        self.gamma = 0.99
        self.camera_urls = [camera_url_0, camera_url_1]
        self.lightbar_url = lightbar_url

    def get_observation(self):
        responses = [requests.get(camera_url) for camera_url in self.camera_urls]
        images = [Image.open(io.BytesIO(response.content)) for response in responses]
        arrays = [np.array(image) for image in images]
        array = np.concatenate(arrays, axis=1)
        return array

    def start(self):
        self.time = time.time()
        self.current_state = self.get_observation()
        return self.current_state

    def step(self, action: np.ndarray):
        requests.put(self.lightbar_url, json={"array": action.tolist()})

        # Define state
        self.current_state = self.get_observation()

        # Compute reward
        self.reward = self.reward_function()

        return self.reward, self.current_state, False, self.get_info()

    def get_info(self):
        return {"gamma": self.gamma}

    def reward_function(self):
        return 0

    # on object destruction
    def __del__(self):
        requests.put(self.lightbar_url, json={"array": np.zeros((2, 6)).tolist()})
