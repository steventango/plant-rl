from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def _load_img_as_tensor(img_pil, image_size):
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


def load_image_frame(
    image: Image.Image,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    images = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    images[0], video_height, video_width = _load_img_as_tensor(image, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def init_state_from_image(
    video_predictor,
    image: Image.Image,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    compute_device = video_predictor.device  # device of the model
    images, video_height, video_width = load_image_frame(
        image=image,
        image_size=video_predictor.image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        compute_device=compute_device,
    )
    inference_state = {}
    inference_state["frame_idx"] = 0
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["frames_tracked_per_obj"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    video_predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


@torch.inference_mode()
def track(
    video_predictor,
    inference_state,
    image,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    """Propagate the input points across frames to track in the entire video."""

    # add a new image frame to the inference state
    compute_device = video_predictor.device  # device of the model
    images, *_ = load_image_frame(
        image=image,
        image_size=video_predictor.image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        compute_device=compute_device,
    )
    inference_state["images"] = images
    inference_state["num_frames"] += 1
    inference_state["frame_idx"] += 1
    frame_idx = inference_state["frame_idx"]

    video_predictor.propagate_in_video_preflight(inference_state)

    obj_ids = inference_state["obj_ids"]
    batch_size = video_predictor._get_obj_num(inference_state)

    pred_masks_per_obj = [None] * batch_size
    for obj_idx in range(batch_size):
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        # We skip those frames already in consolidated outputs (these are frames
        # that received input clicks or mask). Note that we cannot directly run
        # batched forward on them via `_run_single_frame_inference` because the
        # number of clicks on each object might be different.
        if frame_idx in obj_output_dict["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = obj_output_dict[storage_key][frame_idx]
            device = inference_state["device"]
            pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
            if video_predictor.clear_non_cond_mem_around_input:
                # clear non-conditioning memory of the surrounding frames
                video_predictor._clear_obj_non_cond_mem_around_input(
                    inference_state, frame_idx, obj_idx
                )
        else:
            storage_key = "non_cond_frame_outputs"
            current_out, pred_masks = video_predictor._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=obj_output_dict,
                frame_idx=0,
                batch_size=1,  # run on the slice of a single object
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            obj_output_dict[storage_key][frame_idx] = current_out

        inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
            "reverse": False
        }
        pred_masks_per_obj[obj_idx] = pred_masks

    # Resize the output mask to the original video resolution (we directly use
    # the mask scores on GPU for output to avoid any CPU conversion in between)
    if len(pred_masks_per_obj) > 1:
        all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
    else:
        all_pred_masks = pred_masks_per_obj[0]
    _, video_res_masks = video_predictor._get_orig_video_res_output(
        inference_state, all_pred_masks
    )
    return frame_idx, obj_ids, video_res_masks


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

    def inference(self, image: Image.Image, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
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
        self.sam2_predictor = SAM2VideoPredictor.from_pretrained(sam2_model)
        self.state = None

    def inference(self, image: Image.Image, boxes: np.ndarray, multimask_output: bool = False):
        # Setup image for SAM2 and predict masks
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext(),
        ):
            if self.state is None:
                self.state = init_state_from_image(self.sam2_predictor, image)
                for object_id, box in enumerate(boxes, start=1):
                    _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                        inference_state=self.state,
                        frame_idx=0,
                        obj_id=object_id,
                        box=box,
                    )

            # frame_idx, object_ids, masks = self.sam2_predictor.propagate_in_video(self.state, image)
            frame_idx, obj_ids, masks = track(
                video_predictor=self.sam2_predictor,
                inference_state=self.state,
                image=image,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )
            # convert to numpy
            masks = masks.cpu().numpy()
            if masks.ndim == 4:
                masks = masks.squeeze(1)
        return masks, None, None
