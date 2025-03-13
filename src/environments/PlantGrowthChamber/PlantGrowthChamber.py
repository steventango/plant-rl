import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import requests
from PIL import Image

from utils.RlGlue.environment import BaseAsyncEnvironment

from .zones import get_zone


class PlantGrowthChamber(BaseAsyncEnvironment):

    def __init__(self, zone: int, start_time: float | None = None):
        self.gamma = 0.99
        self.zone = get_zone(zone)
        self.images = {}
        self.image = None
        self.time = None
        self._start_time = start_time

    def get_observation(self):
        self.time = datetime.now(tz=ZoneInfo("localtime")).timestamp()

        self.get_image()
        if "left" in self.images and "right" in self.images:
            self.image = np.hstack((np.array(self.images["left"]), np.array(self.images["right"])))
        elif "left" in self.images:
            self.image = np.array(self.images["left"])
        elif "right" in self.images:
            self.image = np.array(self.images["right"])

        self.processed_image = self.process_image(self.image)

        self.plant_stats = np.random.randn(16, 1)

        return self.time, self.image, self.plant_stats

    def get_image(self):

        def fetch_image(url: str):
            response = requests.get(url, timeout=30)
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

    def process_image(self, image: np.ndarray):
        POT_WIDTH = 100
        POT_HEIGHT = 100
        # Get source points from the tray's rectangle
        if self.zone.trays:
            processed_images = []
            for tray in self.zone.trays:
                src_points = np.array(
                    [tray.rect.top_left, tray.rect.top_right, tray.rect.bottom_right, tray.rect.bottom_left],
                    dtype=np.float32,
                )
                # Calculate the height and width from pot widths and heights in each tray
                width = tray.n_wide * POT_WIDTH
                height = tray.n_tall * POT_HEIGHT

                # Define destination points (desired corners of the tray)
                dst_points = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
                )
                 # Compute the homography matrix
                homography_matrix, _ = cv2.findHomography(src_points, dst_points)

                # Warp the image using the homography matrix
                processed_image = cv2.warpPerspective(image, homography_matrix, (width, height))

                processed_images.append(processed_image)

            processed_image = np.hstack(processed_images)
        else:
            processed_image = image

        return processed_image

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
        action = np.tile(action, (2, 1))
        response = requests.put(self.zone.lightbar_url, json={"array": action.tolist()}, timeout=5)
        response.raise_for_status()

    def get_info(self):
        return {"gamma": self.gamma}

    def reward_function(self):
        return np.array(self.image).mean() / 255

    def close(self):
        requests.put(self.zone.lightbar_url, json={"array": np.zeros((2, 6)).tolist()})
