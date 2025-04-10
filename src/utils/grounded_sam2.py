from contextlib import nullcontext

import numpy as np
import torch
from PIL.Image import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundedSAM2:
    def __init__(
        self,
        grounding_model="IDEA-Research/grounding-dino-base",
        sam2_model="facebook/sam2.1-hiera-small",
        device=None,
    ):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Configure environment if using CUDA
        if self.device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 Image Predictor
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model)
        # Load processor and grounding model
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(self.device)

    def inference(self, image: Image, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        """
        Run inference on a single image.

        Args:
            image (Image): Input image.
            text_prompt (str): Text prompt for object detection.
            box_threshold (float): Box threshold for filtering boxes.
            text_threshold (float): Text threshold for filtering boxes.

        Returns:
            boxes (np.ndarray): Bounding boxes of the detected objects.
            masks (np.ndarray): Predicted masks.
            scores (np.ndarray): Scores of the predicted masks.
            logits (np.ndarray): Logits of the predicted masks.
        """
        # Ensure the text prompt is formatted correctly.
        text_prompt = text_prompt.lower()
        if not text_prompt.endswith("."):
            text_prompt += "."

        with (
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
            torch.no_grad(),
        ):
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
            boxes = results[0]["boxes"].cpu().numpy()

            # Setup image for SAM2 and predict masks
            self.sam2_predictor.set_image(np.array(image.convert("RGB")))
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

            if masks.ndim == 4:
                masks = masks.squeeze(1)

        return boxes, masks, scores, logits
