import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import litserve as ls
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SegmentAnythingAPI(ls.LitAPI):
    def setup(self, device, sam2_model="facebook/sam2.1-hiera-small"):
        self.device = device
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
        # Unpack the batch input lists
        images, boxes_list, multimask_outputs = batch_input

        # Process images in batch mode using SAM2's batch capabilities
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
        ):
            # Set image batch for SAM2 predictor
            self.sam2_predictor.set_image_batch(images)

            # Check if any boxes are provided
            has_boxes = any(len(boxes) > 0 for boxes in boxes_list)

            if not has_boxes:
                # No boxes provided for any images, return empty results
                results = [
                    {"masks": np.array([]), "scores": np.array([]), "logits": np.array([])}
                    for _ in range(len(images))
                ]
            else:
                # Use batch prediction
                masks_batch, scores_batch, logits_batch = self.sam2_predictor.predict_batch(
                    point_coords_batch=[None] * len(images),
                    point_labels_batch=[None] * len(images),
                    box_batch=boxes_list,
                    mask_input_batch=[None] * len(images),
                    multimask_output=multimask_outputs[0],  # Using the first value as default
                )

                results = []
                for i in range(len(images)):
                    masks = masks_batch[i]
                    scores = scores_batch[i]
                    logits = logits_batch[i]

                    # Handle case where this specific image had no boxes
                    if len(boxes_list[i]) == 0:
                        results.append({"masks": np.array([]), "scores": np.array([]), "logits": np.array([])})
                    else:
                        # Ensure masks are 3D if they come back as 4D
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)
                        results.append({"masks": masks, "scores": scores, "logits": logits})

            # Reset predictor state after batch processing
            self.sam2_predictor.reset_predictor()

        return results

    def unbatch(self, output):
        return output

    def encode_response(self, result):
        # Convert numpy arrays to lists for JSON serialization
        return {
            "masks": result["masks"].tolist() if len(result["masks"]) > 0 else [],
            "scores": result["scores"].tolist() if len(result["scores"]) > 0 else [],
            "logits": result["logits"].tolist() if len(result["logits"]) > 0 else [],
        }


if __name__ == "__main__":
    api = SegmentAnythingAPI()
    server = ls.LitServer(api, max_batch_size=16, batch_timeout=0.01)
    server.run(port=8000, num_api_servers=4, generate_client_file=False)
