import base64
import io
import logging
import os
import time
import traceback
from contextlib import nullcontext

import cv2
import litserve as ls
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sam_profiling.log"), logging.StreamHandler()],
)
logger = logging.getLogger("sam_api")


class SegmentAnythingAPI(ls.LitAPI):
    def setup(self, device, sam2_model="facebook/sam2.1-hiera-small"):
        setup_start = time.time()
        logger.info(f"Setting up Segment Anything model on device: {device}, model: {sam2_model}")

        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            logger.info(
                f"Using TF32 optimizations on GPU with compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            )
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 Image Predictor
        model_load_start = time.time()
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model)
        model_load_end = time.time()

        logger.info(f"Model loading time: {model_load_end - model_load_start:.4f}s")
        logger.info(f"Total setup time: {model_load_end - setup_start:.4f}s")

    def decode_request(self, request):
        start_time = time.time()
        logger.info("Decoding request")

        image_data = request["image_data"]
        boxes = np.array(request.get("boxes", []))
        multimask_output = request.get("multimask_output", False)

        logger.info(f"Request contains image data ({len(image_data)/1024:.1f}KB), {len(boxes)} boxes")
        logger.info(f"Request decode time: {time.time() - start_time:.4f}s")

        return {
            "image_data": image_data,
            "boxes": boxes,
            "multimask_output": multimask_output,
        }

    def predict(self, input_data):
        start_time = time.time()
        logger.info("Starting prediction")

        # Decode base64 image
        image_data = input_data["image_data"]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Log image size
        image_size = image.size
        logger.info(f"Processing image of size: {image_size}")

        # Get input parameters
        boxes = input_data["boxes"]
        multimask_output = input_data["multimask_output"]
        logger.info(f"Number of boxes: {len(boxes)}, Multimask output: {multimask_output}")

        decode_time = time.time()
        logger.info(f"Image decode time: {decode_time - start_time:.4f}s")

        # Check if boxes are provided
        if len(boxes) == 0:
            logger.info("No boxes provided, returning empty result")
            return {"masks": np.array([]), "scores": np.array([])}

        # Perform inference directly
        set_image_start = time.time()
        try:
            with (
                torch.inference_mode(),
                (
                    torch.autocast(device_type=self.device, dtype=torch.bfloat16)
                    if self.device == "cuda"
                    else nullcontext()
                ),
            ):
                self.sam2_predictor.set_image(np.array(image.convert("RGB")))
                set_image_end = time.time()
                logger.info(f"Set image time: {set_image_end - set_image_start:.4f}s")

                predict_start = time.time()
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=multimask_output,
                )
                predict_end = time.time()
                logger.info(f"Predict time: {predict_end - predict_start:.4f}s")

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                # Log results statistics
                logger.info(f"Generated {len(masks)} masks")
                if len(scores) > 0:
                    logger.info(
                        f"Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}"
                    )

                # Convert masks to contours
                contour_start = time.time()
                contours = []
                for mask in masks:
                    mask = mask.astype("uint8") * 255
                    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    biggest = max(cs, key=cv2.contourArea) if cs else None
                    contour = biggest[:, 0, :].tolist() if biggest is not None else []
                    contours.append(contour)
                contour_end = time.time()
                logger.info(f"Contour extraction time: {contour_end - contour_start:.4f}s")

                # Log the size of original response data for comparison
                original_masks_size_mb = masks.nbytes / (1024 * 1024)
                scores_size_mb = scores.nbytes / (1024 * 1024)
                logits_size_mb = logits.nbytes / (1024 * 1024)
                total_original_size_mb = original_masks_size_mb + scores_size_mb + logits_size_mb

                logger.info(
                    f"Original response data sizes - Masks: {original_masks_size_mb:.2f}MB, Scores: {scores_size_mb:.2f}MB, Logits: {logits_size_mb:.2f}MB"
                )
                logger.info(f"Total original response data size: {total_original_size_mb:.2f}MB")

                # Log the size of new contours response data
                import sys

                contours_size_mb = sys.getsizeof(str(contours)) / (1024 * 1024)
                new_total_size_mb = contours_size_mb + scores_size_mb

                logger.info(
                    f"New response data sizes - Contours: {contours_size_mb:.2f}MB, Scores: {scores_size_mb:.2f}MB"
                )
                logger.info(f"Total new response data size: {new_total_size_mb:.2f}MB")

                # Log bandwidth reduction
                size_reduction_mb = total_original_size_mb - new_total_size_mb
                reduction_percentage = (
                    (size_reduction_mb / total_original_size_mb) * 100 if total_original_size_mb > 0 else 0
                )

                logger.info(f"Bandwidth reduction: {size_reduction_mb:.2f}MB ({reduction_percentage:.2f}%)")

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return {"contours": [], "scores": np.array([])}

        end_time = time.time()
        logger.info(f"Total prediction time: {end_time - start_time:.4f}s")

        return {"contours": contours, "scores": scores}

    def encode_response(self, result):
        # Log time taken to encode the response
        encode_start = time.time()

        # Convert numpy arrays to lists for JSON serialization
        response = {
            "contours": result["contours"],
            "scores": result["scores"].tolist() if len(result["scores"]) > 0 else [],
        }

        encode_end = time.time()
        logger.info(f"Response encoding time: {encode_end - encode_start:.4f}s")

        # Estimate the serialized JSON size (rough approximation)
        try:
            import sys

            response_size_mb = sys.getsizeof(str(response)) / (1024 * 1024)
            logger.info(f"Estimated serialized response size: {response_size_mb:.2f}MB")
        except Exception as e:
            logger.warning(f"Could not estimate response size: {str(e)}")

        return response


if __name__ == "__main__":
    api = SegmentAnythingAPI()
    server = ls.LitServer(api, batch_timeout=0.01)
    server.run(port=8000, num_api_servers=1, generate_client_file=False)
