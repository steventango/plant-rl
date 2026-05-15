import base64
import io
import logging
import os
from pathlib import Path


import aiohttp
import numpy as np
from aiohttp_retry import RetryClient
from PIL import Image

logger = logging.getLogger("plant_rl.CVPipelineClient")

PIPELINE_URL = os.getenv("PIPELINE_URL", "http://localhost:8800")


class CVPipelineClient:
    def __init__(self, dataset_path: Path | None = None):
        self.set_dataset_path(dataset_path)

    def set_dataset_path(self, path: Path | None = None):
        self.dataset_path = path
        if self.dataset_path:
            (self.dataset_path / "visualization").mkdir(parents=True, exist_ok=True)

    async def detect(
        self,
        session: aiohttp.ClientSession | RetryClient,
        image: np.ndarray | Image.Image,
    ):
        """
        Initializes tracking by calling the pipeline detect endpoint.
        Returns the full JSON response from the pipeline.
        """
        try:
            image_data = encode_image(image)
            payload = {
                "image_data": image_data,
            }
            # Bounded per-call so the cv_session default (45 s) doesn't let a
            # stalled detect chew through get_observation's 50 s wait_for budget.
            async with session.post(
                f"{PIPELINE_URL}/pipeline/detect",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=35),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.exception(f"Error during pot detection: {e}")
            return None

    async def propagate(
        self,
        session: aiohttp.ClientSession | RetryClient,
        image: np.ndarray | Image.Image,
        state: dict | str,
    ):
        """
        Propagates tracking to the next frame.
        Returns the full JSON response from the pipeline.
        """
        try:
            image_data = encode_image(image)
            payload = {
                "image_data": image_data,
                "state": state,
            }
            async with session.post(
                f"{PIPELINE_URL}/pipeline/propagate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=35),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.exception(f"Error during propagation: {e}")
            return None

    def _save_image(self, b64_str, name):
        if not self.dataset_path:
            return

        try:
            img_data = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_data))
            path = self.dataset_path / "visualization" / f"{name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
        except Exception as e:
            logger.warning(f"Failed to save visualization {name}: {e}")


def encode_image(image_array):
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array.astype(np.uint8))
    else:
        image = image_array

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
