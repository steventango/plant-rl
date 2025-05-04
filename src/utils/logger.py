import numpy as np
import pandas as pd

import wandb


def expand(key, value):
    if isinstance(value, np.ndarray):
        result = {}
        for idx in np.ndindex(value.shape):
            idx_str = ".".join(map(str, idx))
            result[f"{key}.{idx_str}"] = value[idx]
        return result
    if isinstance(value, (list, tuple)):
        return {f"{key}.{i}": v for i, v in enumerate(value)}
    else:
        return {key: value}


def format_bounding_boxes(detections):
    """Format bounding boxes for wandb visualization.

    Args:
        detections: Detection object containing xyxy, confidence, and class_id

    Returns:
        box_data: List of box entries
        class_id_to_label: Dictionary mapping class IDs to labels
    """
    box_data = []
    class_id_to_label = {}

    if detections.xyxy is not None and len(detections.xyxy) > 0:
        for i, (box, conf, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
            # Store class ID to label mapping
            class_id_to_label[int(class_id)] = str(class_id)

            # Create box entry
            box_entry = {
                "position": {
                    "minX": float(box[0]),
                    "minY": float(box[1]),
                    "maxX": float(box[2]),
                    "maxY": float(box[3]),
                },
                "domain": "pixel",
                "class_id": int(class_id),
                "box_caption": f"{class_id}: {conf:.2f}",
                "scores": {"confidence": float(conf)},
            }
            box_data.append(box_entry)

    return box_data, class_id_to_label


def format_masks(detections, class_id_to_label):
    """Format segmentation masks for wandb visualization.

    Args:
        detections: Detection object containing mask and class_id
        class_id_to_label: Dictionary mapping class IDs to labels

    Returns:
        masks_dict: Dictionary containing mask data and class labels
    """
    masks_dict = {}

    if hasattr(detections, "mask") and detections.mask is not None and detections.mask.shape[0] > 0:
        if detections.mask.ndim == 3:  # (n, h, w)
            # Create a single mask where each pixel has the value of its class
            mask_data = np.empty(detections.mask.shape[1:], dtype=np.uint8)
            for class_mask, class_id in zip(detections.mask, detections.class_id):
                # Only update pixels that are part of this mask and weren't set by higher-confidence masks
                mask_data = np.where(class_mask, min(class_id, 255), mask_data)

            masks_dict["predictions"] = {"mask_data": mask_data, "class_labels": class_id_to_label}

    return masks_dict


def create_annotated_image(image_data, box_data, class_id_to_label, masks_dict):
    """Create a wandb Image with bounding boxes and masks annotations.

    Args:
        image_data: The image to annotate
        box_data: List of box entries
        class_id_to_label: Dictionary mapping class IDs to labels
        masks_dict: Dictionary containing mask data and class labels

    Returns:
        wandb.Image: Image with annotations
    """
    if box_data or masks_dict:
        return wandb.Image(
            image_data,
            boxes=({"predictions": {"box_data": box_data, "class_labels": class_id_to_label}} if box_data else None),
            masks=masks_dict if masks_dict else None,
            file_type="jpg",
        )
    return wandb.Image(image_data, file_type="jpg")


def log(env, glue, wandb_run, s, a, info, r=None):
    expanded_info = {}
    for key, value in info.items():
        if isinstance(value, pd.DataFrame):
            table = wandb.Table(dataframe=value)
            expanded_info.update({key: table})
        elif isinstance(value, np.ndarray):
            if value.size < 16:
                expanded_info.update(expand(key, value))
        else:
            expanded_info.update(expand(key, value))
    data = {
        **expand("state", s),
        **expand("agent_action", a),
        "steps": glue.num_steps,
        **expanded_info,
    }
    if hasattr(env, "last_action"):
        data.update(expand("action", env.last_action))
    if hasattr(env, "time"):
        data["time"] = env.time.timestamp()
    if hasattr(env, "image"):
        data["raw_image"] = wandb.Image(env.image, file_type="jpg")

        if hasattr(env, "detections"):
            detections = env.detections

            # Extract box and mask data
            box_data, class_id_to_label = format_bounding_boxes(detections)
            masks_dict = format_masks(detections, class_id_to_label)

            # Get the image data to be annotated
            image_data = env.images.get("warped", env.image) if hasattr(env, "images") else env.image

            # Create annotated image
            if box_data or masks_dict:
                data["image"] = create_annotated_image(image_data, box_data, class_id_to_label, masks_dict)

    if r is not None:
        data["reward"] = r
    wandb_run.log(data)
