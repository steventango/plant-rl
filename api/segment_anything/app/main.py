import base64
import io
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
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
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def decode_request(self, request):
        image_data = request["image_data"]
        boxes = np.array(request.get("boxes", []))
        multimask_output = request.get("multimask_output", False)

        return {
            "image_data": image_data,
            "boxes": boxes,
            "multimask_output": multimask_output,
        }

    def batch(self, inputs):
        # Define a function to process each input in parallel
        def process_input(item):
            # Decode base64 image
            image_data = item["image_data"]
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Convert PIL image to numpy array for batch processing
            np_image = np.array(pil_image)

            return {
                "image": np_image,
                "boxes": item["boxes"],
                "multimask_output": item["multimask_output"],
            }

        # Process inputs in parallel using the thread pool
        processed_items = list(self.pool.map(process_input, inputs))

        # Extract the results into separate lists
        images = [item["image"] for item in processed_items]
        boxes_list = [item["boxes"] for item in processed_items]
        multimask_outputs = [item["multimask_output"] for item in processed_items]

        # Return lists directly
        return images, boxes_list, multimask_outputs

    def predict(self, batch_input):
        # If input is not a batch, use the single image processing logic
        if not isinstance(batch_input, tuple):
            return self._predict_single(batch_input)

        # Unpack the batch input lists
        images, boxes_list, multimask_outputs = batch_input

        # Process images in batch mode using SAM2's batch capabilities
        try:
            with (
                torch.inference_mode(),
                (
                    torch.autocast(device_type=self.device, dtype=torch.bfloat16)
                    if self.device == "cuda"
                    else nullcontext()
                ),
            ):
                # Set image batch for SAM2 predictor
                self.sam2_predictor.set_image_batch(images)

                # Check if any boxes are provided
                has_boxes = any(len(boxes) > 0 for boxes in boxes_list)

                if not has_boxes:
                    # No boxes provided for any images, return empty results
                    results = [{"contours": [], "scores": np.array([])} for _ in range(len(images))]
                else:
                    # Use batch prediction
                    masks_batch, scores_batch, logits_batch = self.sam2_predictor.predict_batch(
                        point_coords_batch=[None] * len(images),
                        point_labels_batch=[None] * len(images),
                        box_batch=boxes_list,
                        mask_input_batch=[None] * len(images),
                        multimask_output=any(multimask_outputs),
                    )

                    results = []
                    for i in range(len(images)):
                        masks = masks_batch[i]
                        scores = scores_batch[i]

                        # Handle case where this specific image had no boxes
                        if len(boxes_list[i]) == 0:
                            results.append({"contours": [], "scores": np.array([])})
                        else:
                            # Ensure masks are 3D if they come back as 4D
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

                            results.append({"contours": contours, "scores": scores})

                # Reset predictor state after batch processing
                self.sam2_predictor.reset_predictor()

        except Exception:
            # If there's an error in batch processing, return empty results
            results = [{"contours": [], "scores": np.array([])} for _ in range(len(images))]

        return results

    def _predict_single(self, input_data):
        """Process a single image input (non-batch mode)"""
        # Decode base64 image
        image_data = input_data["image_data"]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get input parameters
        boxes = input_data["boxes"]
        multimask_output = input_data["multimask_output"]

        # Check if boxes are provided
        if len(boxes) == 0:
            return {"contours": [], "scores": np.array([])}

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

                masks, scores, _ = self.sam2_predictor.predict(
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

    def unbatch(self, output):
        return output

    def encode_response(self, result):
        # Convert numpy arrays to lists for JSON serialization
        response = {
            "contours": result["contours"],
            "scores": result["scores"].tolist() if len(result["scores"]) > 0 else [],
        }

        return response


if __name__ == "__main__":
    api = SegmentAnythingAPI()
    server = ls.LitServer(api, max_batch_size=4, batch_timeout=0.01)
    server.run(port=8000, num_api_servers=2, generate_client_file=False)
