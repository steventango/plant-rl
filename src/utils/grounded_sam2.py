from contextlib import nullcontext

import numpy as np
import torch
from PIL.Image import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


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


class SAM2:
    def __init__(
        self,
        sam2_model="facebook/sam2.1-hiera-small",
        device=None,
    ):
        self.device = set_device(device)
        # Build SAM2 Image Predictor
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model)

    def inference(
        self, image: Image, boxes: np.ndarray, multimask_output: bool = False
    ):
        # Setup image for SAM2 and predict masks
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext(),
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
