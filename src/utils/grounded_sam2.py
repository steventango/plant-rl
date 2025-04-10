import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
TEXT_PROMPT = "plant."
IMG_PATH = "tmp/baseline/2025-04-02T165154_warped.jpg"
SAM2_MODEL = "facebook/sam2.1-hiera-small"

device = "cuda" if torch.cuda.is_available() else "cpu"

# environment settings
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL)

    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(device)

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT
    img_path = IMG_PATH

    image = Image.open(img_path)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]]
    )
    input_boxes = results[0]["boxes"].cpu().numpy()

    sam2_predictor.set_image(np.array(image.convert("RGB")))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
