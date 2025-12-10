import base64
import io
import logging
import os
from pathlib import Path

import aiohttp
import numpy as np
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

    async def detect_pots(
        self, session: aiohttp.ClientSession, image: np.ndarray | Image.Image
    ):
        """
        Detects pots in the image.
        Returns a list of quadrilaterals (or None if failed/empty).
        """
        try:
            image_data = encode_image(image)
            visualize = self.dataset_path is not None
            # Detect
            detect_result = await self._call_pipeline(
                session, "pot/detect", {
                "image_data": image_data,
                "visualize": visualize,
            }
            )
            boxes = detect_result["boxes"]

            if self.dataset_path and "visualization" in detect_result:
                self._save_image(detect_result["visualization"], "detect_pots")

            if not boxes:
                logger.warning("No pots detected.")
                return None

            # Segment
            segment_result = await self._call_pipeline(
                session, "pot/segment", {
                "image_data": image_data,
                "boxes": boxes,
                "visualize": visualize,
            }
            )
            masks_b64 = segment_result["masks"]

            if self.dataset_path and "visualization" in segment_result:
                self._save_image(segment_result["visualization"], "segment_pots")

            # Quad
            quad_result = await self._call_pipeline(session, "pot/quad", {
                "masks": masks_b64,
                "visualize": visualize,
                "image_data": image_data,
            })

            if self.dataset_path and "visualization" in quad_result:
                self._save_image(quad_result["visualization"], "quad_pots")

            quads = quad_result["quadrilaterals"]
            logger.info(f"Successfully detected {len(quads)} pots.")
            return quads

        except Exception as e:
            logger.exception(f"Error during pot detection: {e}")
            return None

    async def process_plants(
        self,
        session: aiohttp.ClientSession,
        image: np.ndarray | Image.Image,
        pot_quads: list,
        timestamp_str: str = "",
    ):
        """
        Processes plants within the given pot quadrilaterals.
        Returns a list of stats dictionaries.
        """
        try:
            image_data = encode_image(image)

            # Warp
            warp_result = await self._call_pipeline(session, "pot/warp", {
                "image_data": image_data,
                "quadrilaterals": pot_quads,
                "margin": 0.25,
                "output_size": 256,
            })
            warped_images = warp_result["warped_images"]

            plant_stats_list = []

            for i, warped in enumerate(warped_images):
                if not warped:
                    plant_stats_list.append({})
                    continue

                # Detect Plant
                detect_result = await self._call_pipeline(
                    session, "plant/detect", {"image_data": warped}
                )
                boxes = detect_result["boxes"]

                stats = None
                masks_b64 = None

                if boxes:
                    # Segment Plant
                    segment_result = await self._call_pipeline(
                        session, "plant/segment", {
                        "image_data": warped,
                        "boxes": boxes,
                        "confidences": detect_result["confidences"],
                    }
                    )

                    if segment_result["success"]:
                        masks_b64 = segment_result.get("masks")
                        # Stats
                        stats_result = await self._call_pipeline(session, "plant/stats", {
                            "warped_image": warped,
                            "mask": segment_result["mask"],
                            "pot_size_mm": 60.0,
                            "margin": 0.25,
                        })
                        stats = stats_result["stats"]

                if stats:
                    plant_stats_list.append(stats)
                else:
                    plant_stats_list.append({})

                # Visualization (only if requested)
                if self.dataset_path:
                    vis_result = await self._call_pipeline(
                        session, "plant/visualize",
                        {
                            "image_data": warped,
                            "boxes": boxes,
                            "confidences": detect_result.get("confidences", []),
                            "masks": masks_b64,
                            "stats": stats is not None,
                            "selected_index": segment_result.get("selected_index")
                            if boxes
                            else None,
                            "mask_scores": segment_result.get("mask_scores")
                            if boxes
                            else None,
                            "combined_scores": segment_result.get("combined_scores")
                            if boxes
                            else None,
                            "pot_size_mm": 60.0,
                            "margin": 0.25,
                        }
                    )
                    if "visualization" in vis_result:
                        self._save_image(
                            vis_result["visualization"], f"{timestamp_str}/{i:02d}"
                        )

            return plant_stats_list

        except Exception as e:
            logger.exception(f"Error during plant processing: {e}")
            return []

    async def _call_pipeline(self, session, endpoint, payload):
        async with session.post(f"{PIPELINE_URL}/{endpoint}", json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

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
