import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import litserve as ls
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingDinoAPI(ls.LitAPI):
    def setup(self, device, pretrained_model_name_or_path="IDEA-Research/grounding-dino-base"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def decode_request(self, request):
        image_data = request["image_data"]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        text_prompt = request["text_prompt"]
        text_prompt = text_prompt.lower()
        if not text_prompt.endswith("."):
            text_prompt += "."
        box_threshold = request.get("box_threshold", 0.3)
        text_threshold = request.get("text_threshold", 0.25)
        return {
            "image": image,
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }

    def predict(self, x, **kwargs):
        # Process all images in a single batch, but handle post-processing individually
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
        ):
            print(x)
            image = x["image"]
            text_prompt = x["text_prompt"]
            box_threshold = x["box_threshold"]
            text_threshold = x["text_threshold"]
            target_sizes = [image.size[::-1]]
            # Process all images with their respective text prompts in a single batch
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            return self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]

    def encode_response(self, detection):
        return {
            "boxes": detection["boxes"].cpu().numpy().tolist(),
            "scores": detection["scores"].cpu().numpy().tolist(),
            "text_labels": detection["text_labels"],
            "labels": detection["labels"],
        }


if __name__ == "__main__":
    api = GroundingDinoAPI()
    server = ls.LitServer(api, accelerator="gpu")
    server.run(port=8000, num_api_servers=4)
