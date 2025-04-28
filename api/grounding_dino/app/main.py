import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import litserve as ls
import torch
from PIL import Image
from processing_grounding_dino import BatchGroundingDinoProcessor
from transformers import AutoModelForZeroShotObjectDetection


class GroundingDinoAPI(ls.LitAPI):
    def setup(self, device, pretrained_model_name_or_path="IDEA-Research/grounding-dino-base"):
        self.device = device
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.processor = BatchGroundingDinoProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def decode_request(self, request):
        image_data = request["image_data"]
        text_prompt = request["text_prompt"]
        text_prompt = text_prompt.lower()
        if not text_prompt.endswith("."):
            text_prompt += "."
        threshold = request.get("threshold", 0.3)
        text_threshold = request.get("text_threshold", 0.25)
        return {
            "image_data": image_data,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "text_threshold": text_threshold,
        }

    def batch(self, inputs):
        # Define a function to process each input in parallel
        def process_input(item):
            # Decode base64 image
            image_data = item["image_data"]
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            return {
                "image": pil_image,
                "text_prompt": item["text_prompt"],
                "threshold": item["threshold"],
                "text_threshold": item["text_threshold"],
                "target_size": pil_image.size[::-1],
            }

        # Process inputs in parallel using the thread pool
        processed_items = list(self.pool.map(process_input, inputs))

        # Extract the results into separate lists
        images = [item["image"] for item in processed_items]
        text_prompts = [item["text_prompt"] for item in processed_items]
        thresholds = [item["threshold"] for item in processed_items]
        text_thresholds = [item["text_threshold"] for item in processed_items]
        target_sizes = [item["target_size"] for item in processed_items]

        # Return lists directly
        return images, text_prompts, thresholds, text_thresholds, target_sizes

    def predict(self, batch_input):
        # Unpack the batch input lists
        images, text_prompts, thresholds, text_thresholds, target_sizes = batch_input

        # Process all images in a single batch, but handle post-processing individually
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
        ):
            # Process all images with their respective text prompts in a single batch
            inputs = self.processor(images=images, text=text_prompts, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            return self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                thresholds=thresholds,
                text_thresholds=text_thresholds,
                target_sizes=target_sizes,
            )

    def unbatch(self, output):
        return output

    def encode_response(self, detection):
        return {
            "boxes": detection["boxes"].cpu().numpy().tolist(),
            "scores": detection["scores"].cpu().numpy().tolist(),
            "text_labels": detection["text_labels"],
        }


if __name__ == "__main__":
    api = GroundingDinoAPI()
    server = ls.LitServer(api, max_batch_size=16, batch_timeout=0.01)
    server.run(port=8000, num_api_servers=4, generate_client_file=False)
