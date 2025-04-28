import base64
import io
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


class SegmentAnythingAPI(ls.LitAPI):

    def setup(self, device, sam2_model="facebook/sam2.1-hiera-small"):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 Image Predictor
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model)

    def decode_request(self, request):
        image_data = request["image_data"]
        boxes = np.array(request.get("boxes", []))
        multimask_output = request.get("multimask_output", False)

        return {
            "image_data": image_data,
            "boxes": boxes,
            "multimask_output": multimask_output,
        }

    def predict(self, input_data):
        # Decode base64 image
        image_data = input_data["image_data"]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get input parameters
        boxes = input_data["boxes"]
        multimask_output = input_data["multimask_output"]

        # Check if boxes are provided
        if len(boxes) == 0:
            return {"masks": np.array([]), "scores": np.array([])}

        # Perform inference directly
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

                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=multimask_output,
                )

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                # Convert masks to contours
                contours = []
                for mask in masks:
                    mask = mask.astype("uint8") * 255
                    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    biggest = max(cs, key=cv2.contourArea) if cs else None
                    contour = biggest[:, 0, :].tolist() if biggest is not None else []
                    contours.append(contour)

        except Exception:
            return {"contours": [], "scores": np.array([])}

        return {"contours": contours, "scores": scores}

    def encode_response(self, result):
        # Convert numpy arrays to lists for JSON serialization
        response = {
            "contours": result["contours"],
            "scores": result["scores"].tolist() if len(result["scores"]) > 0 else [],
        }

        return response


if __name__ == "__main__":
    api = SegmentAnythingAPI()
    server = ls.LitServer(api, batch_timeout=0.01)
    server.run(port=8000, num_api_servers=1, generate_client_file=False)
