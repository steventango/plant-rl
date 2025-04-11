from contextlib import nullcontext

import numpy as np
import torch
from PIL.Image import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def set_device(device):
    if device is not None:
        device = device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device


class GroundingDino:
    def __init__(
        self,
        pretrained_model_name_or_path="IDEA-Research/grounding-dino-base",
        device=None,
    ):
        self.device = set_device(device)

        # Load processor and grounding model
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(pretrained_model_name_or_path).to(self.device)

    def inference(self, image: Image, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        # Process text prompt and run grounding detection
        text_prompt = text_prompt.lower()
        if not text_prompt.endswith("."):
            text_prompt += "."

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
        ):
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
            boxes = results[0]["boxes"].cpu().numpy()
            confidences = results[0]["scores"].cpu().numpy()
            class_names = np.array(results[0]["text_labels"])

        return boxes, confidences, class_names


class SAM2:
    def __init__(
        self,
        sam2_model="facebook/sam2.1-hiera-small",
        device=None,
    ):
        self.device = set_device(device)
        # Build SAM2 Image Predictor
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model)

    def inference(self, image: Image, boxes: np.ndarray, multimask_output: bool = False):
        # Setup image for SAM2 and predict masks
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
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
        return masks, scores, logits
