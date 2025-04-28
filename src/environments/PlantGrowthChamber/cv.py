import base64
import io
import json
from math import ceil

import cv2
import numpy as np
import pandas as pd
import requests
import supervision as sv
from PIL import Image
from plantcv import plantcv as pcv
from plantcv.plantcv import params
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from supervision.draw.color import DEFAULT_COLOR_PALETTE, ColorPalette

from .zones import POT_HEIGHT, POT_WIDTH, SCALE, Tray

CUSTOM_COLOR_PALETTE = DEFAULT_COLOR_PALETTE * ceil(128 / len(DEFAULT_COLOR_PALETTE)) + ["#808080"] * 1000
color_palette_custom = ColorPalette.from_hex(CUSTOM_COLOR_PALETTE)


def call_segment_anything_api(image, boxes, multimask_output=False, server_url="http://segment-anything:8000/predict"):
    """
    Call the Segment Anything API with an image and bounding boxes

    Args:
        image: PIL Image to be processed
        boxes: Numpy array of boxes in format [x1, y1, x2, y2]
        multimask_output: Whether to return multiple masks per box
        server_url: URL of the Segment Anything API server

    Returns:
        masks: Numpy array of binary masks (converted from contours)
        scores: Numpy array of confidence scores for each mask
    """
    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Prepare the payload
    payload = json.dumps(
        {
            "image_data": img_str,
            "boxes": boxes.tolist(),
            "multimask_output": multimask_output,
        }
    )

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(server_url, data=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Convert contours to masks
        image_height, image_width = np.array(image).shape[:2]
        masks = np.zeros((len(result["contours"]), image_height, image_width), dtype=np.uint8)

        for i, contour in enumerate(result["contours"]):
            if contour:
                # Convert contour to numpy array with correct shape for fillPoly
                contour_np = np.array(contour, dtype=np.int32)
                # Create mask from contour
                cv2.fillPoly(masks[i], [contour_np], 1)

        # Convert lists back to numpy arrays
        scores = np.array(result["scores"])

        return masks, scores
    except Exception as e:
        print(f"Error calling Segment Anything API: {e}")
        return np.array([]), np.array([])


def call_grounding_dino_api(
    image, text_prompt, threshold=0.05, text_threshold=0.05, server_url="http://grounding-dino:8000/predict"
):
    """
    Call the Grounding DINO API with an image and text prompt

    Args:
        image: PIL Image to be processed
        text_prompt: Text prompt for object detection
        threshold: Confidence threshold for boxes
        text_threshold: Text threshold
        server_url: URL of the Grounding DINO API server

    Returns:
        boxes: Numpy array of boxes in format [x1, y1, x2, y2]
        confidences: Numpy array of confidence scores
        class_names: List of detected class names
    """

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Prepare the payload
    payload = json.dumps(
        {
            "image_data": img_str,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "text_threshold": text_threshold,
        }
    )

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(server_url, data=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        boxes = np.array(result["boxes"])
        confidences = np.array(result["scores"])
        class_names = result["text_labels"]

        return boxes, confidences, class_names
    except Exception as e:
        print(f"Error calling Grounding DINO API: {e}")
        return np.array([]), np.array([]), []


def process_image(image: np.ndarray, trays: list[Tray], debug_images: dict[str, Image.Image]):
    if not trays:
        raise ValueError("No trays provided")

    camera_matrix = np.array([[1800.0, 0.0, 1296.0], [0.0, 1800.0, 972.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
    # convert image from RGBA to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    debug_images["undistorted"] = Image.fromarray(undistorted_image)

    # infer_pot_positions(trays, debug_images, undistorted_image)

    # WARP image and recombine
    warped_tray_images = []
    for tray in trays:
        warped_tray_image = warp(undistorted_image, tray)
        warped_tray_images.append(warped_tray_image)
    images = warped_tray_images
    if len(images) > 1:
        # stack images on the longest axis
        if images[0].shape[0] > images[0].shape[1]:
            warped_image = np.hstack(images)
            n_tall = max(tray.n_tall for tray in trays)
            n_wide = sum(tray.n_wide for tray in trays)
        else:
            warped_image = np.vstack(images)
            n_tall = sum(tray.n_tall for tray in trays)
            n_wide = max(tray.n_wide for tray in trays)
    else:
        warped_image = images[0]
        n_tall = trays[0].n_tall
        n_wide = trays[0].n_wide
    debug_images["warped"] = Image.fromarray(warped_image)

    pot_width = POT_WIDTH
    num_plants = n_tall * n_wide
    height, width = warped_image.shape[:2]
    pil_image = Image.fromarray(warped_image)

    gray_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
    debug_images["gray"] = Image.fromarray(gray_image)

    gray_image = gray_image.astype(np.uint8)
    equalized = equalize_adapthist(gray_image, kernel_size=pot_width, clip_limit=0.01)
    equalized = (equalized * 255).astype(np.uint8)

    # Replace direct grounding_dino.inference() call with API call
    boxes, confidences, class_names = call_grounding_dino_api(
        image=pil_image,
        text_prompt="plant.",
        threshold=0.05,
        text_threshold=0.05,
    )

    class_ids = np.full(len(class_names), 901, dtype=int)
    detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)

    # filter out boxes that are bigger than the pot size
    bigger_ratio = 1.5
    size_filter = (boxes[:, 2] - boxes[:, 0] < pot_width * bigger_ratio) & (
        boxes[:, 3] - boxes[:, 1] < pot_width * bigger_ratio
    )

    detections.class_id[~size_filter] = 903

    coords = []
    sigma = 1.5 * pot_width
    for j in range(n_tall):
        for i in range(n_wide):
            y = j * pot_width + pot_width // 2
            x = i * pot_width + pot_width // 2
            coords.append((x, y))

            # box is in range if box center is inside the pot
            box_centers = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            boxes_in_range = (
                (box_centers[0] > x - pot_width // 2)
                & (box_centers[0] < x + pot_width // 2)
                & (box_centers[1] > y - pot_width // 2)
                & (box_centers[1] < y + pot_width // 2)
            )

            # RBF score for boxes closer to the center of the pot
            score = np.exp(
                -((x - boxes[boxes_in_range, 0]) ** 2 + (y - boxes[boxes_in_range, 1]) ** 2) / (sigma / 2) ** 2
            )
            confidences[boxes_in_range] *= score

            # only keep the box with the highest confidence
            if boxes_in_range.nonzero()[0].size < 2:
                continue

            # assign class_ids to the box with the highest confidence
            max_confidence_index = boxes_in_range.nonzero()[0][np.argmax(confidences[boxes_in_range])]
            detections.class_id[max_confidence_index] = i + j * n_wide

    detections = detections.with_nms(threshold=0.01)
    detections.class_id[detections.confidence < 0.05] = 902
    reason_codes = {
        901: "relative low confidence",
        902: "low confidence",
        903: "too big",
    }

    box_annotator = sv.BoxAnnotator(color_palette_custom)
    annotate_detections = detections[np.argsort(detections.class_id)[::-1]]
    annotated_frame = box_annotator.annotate(scene=warped_image.copy(), detections=annotate_detections)

    label_annotator = sv.LabelAnnotator(color_palette_custom)
    labels = [
        f"{'R' if class_id in reason_codes else ''}{class_id} {confidence:.2f}"
        for class_id, confidence in zip(annotate_detections.class_id, annotate_detections.confidence)
    ]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=annotate_detections, labels=labels)
    debug_images["boxes"] = Image.fromarray(annotated_frame)

    masks = np.zeros((len(detections), height, width), dtype=bool)
    valid_detections = detections[detections.class_id < 901]
    if not len(valid_detections):
        columns = [
            "in_bounds",
            "area",
            "convex_hull_area",
            "solidity",
            "perimeter",
            "width",
            "height",
            "longest_path",
            "center_of_mass_x",
            "center_of_mass_y",
            "convex_hull_vertices",
            "object_in_frame",
            "ellipse_center_x",
            "ellipse_center_y",
            "ellipse_major_axis",
            "ellipse_minor_axis",
            "ellipse_angle",
            "ellipse_eccentricity",
        ]
        return pd.DataFrame([{**{col: 0 for col in columns}, "plant_id": i + 1} for i in range(num_plants)])
    new_masks, *_ = call_segment_anything_api(
        image=pil_image,
        boxes=valid_detections.xyxy,
    )
    masks[detections.class_id < 901] = new_masks.astype(bool)

    detections.mask = masks

    mask_areas = np.sum(detections.mask, axis=(1, 2))
    convex_hull = []
    for mask in detections.mask:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            convex_hull.append(hull)
        else:
            convex_hull.append(np.array([]))
    convex_hull_areas = np.array([cv2.contourArea(hull) if hull.size > 0 else 0 for hull in convex_hull])
    size_filter = (mask_areas > 0.7 * pot_width * pot_width) & (mask_areas > 0.95 * convex_hull_areas)
    detections.class_id[size_filter] = 1003

    mask_annotator = sv.MaskAnnotator(color_palette_custom)
    annotate_detections = detections[(detections.class_id < 901) | (detections.class_id > 1000)]
    # sort by class_id
    annotate_detections = annotate_detections[np.argsort(annotate_detections.class_id)[::-1]]
    annotated_frame = mask_annotator.annotate(scene=warped_image.copy(), detections=annotate_detections)
    debug_images["masks"] = Image.fromarray(annotated_frame)

    label_annotator = sv.LabelAnnotator(color_palette_custom)
    labels = [
        f"{'R' if class_id in reason_codes else ''}{class_id} {confidence:.2f}"
        for class_id, confidence in zip(annotate_detections.class_id, annotate_detections.confidence)
    ]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=annotate_detections, labels=labels)

    debug_images["equalized"] = Image.fromarray(equalized)
    otsu_threshold = threshold_otsu(equalized[detections.mask.any(axis=0)].reshape((1, -1)))
    mask = pcv.threshold.binary(equalized, threshold=otsu_threshold, object_type="light")
    debug_images["mask"] = Image.fromarray(mask)

    detections.mask &= mask.astype(bool)

    mask_annotator = sv.MaskAnnotator(color_palette_custom)
    annotate_detections = detections[(detections.class_id < 901) | (detections.class_id > 1000)]
    # sort by class_id
    annotate_detections = annotate_detections[np.argsort(annotate_detections.class_id)[::-1]]
    annotated_frame = mask_annotator.annotate(scene=warped_image.copy(), detections=annotate_detections)

    labeled_mask = detections.mask.astype(np.uint8) * (detections.class_id[:, None, None] + 1)
    labeled_mask[labeled_mask > 900] = 0
    labeled_mask = np.sum(labeled_mask, axis=0)

    params.line_thickness = 1

    shape_image = pcv.analyze.size(img=warped_image, labeled_mask=labeled_mask, n_labels=num_plants)
    debug_images["shape_image"] = Image.fromarray(shape_image)

    stats = []
    area_map = {}
    for sample, variables in pcv.outputs.observations.items():
        row = {}
        plant_num = int(sample.removeprefix("default_")) - 1
        row["plant_id"] = plant_num

        for variable, value in variables.items():
            if variable == "center_of_mass":
                row["center_of_mass_x"], row["center_of_mass_y"] = value["value"]
            elif variable == "ellipse_center":
                row["ellipse_center_x"], row["ellipse_center_y"] = value["value"]
            else:
                row[variable] = value["value"]
        row["area"] /= SCALE**2
        area_map[plant_num] = row["area"]
        stats.append(row)

    # convert all_plant_stats to pandas dataframe
    df = pd.DataFrame(stats)

    # annotate with area
    labels = [f"{plant_id}: {area_map.get(plant_id, 0):.2f} mm²" for plant_id in annotate_detections.class_id]
    label_annotator = sv.RichLabelAnnotator(
        color_palette_custom,
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans",
        font_size=16,
    )
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=annotate_detections, labels=labels)

    debug_images["masks2"] = Image.fromarray(annotated_frame)
    return df


def infer_pot_positions(trays, debug_images, undistorted_image):
    boxes, confidences, class_names = call_grounding_dino_api(
        image=Image.fromarray(undistorted_image),
        text_prompt="pot with soil.",
        threshold=0.05,
        text_threshold=0.05,
    )

    num_pots_wide = sum(tray.n_wide for tray in trays)
    num_pots_tall = sum(tray.n_tall for tray in trays)
    size_filter = (boxes[:, 2] - boxes[:, 0] < undistorted_image.shape[1] / num_pots_wide * 1.2) & (
        boxes[:, 3] - boxes[:, 1] < undistorted_image.shape[0] / num_pots_tall * 1.2
    )

    boxes = boxes[size_filter]
    confidences = confidences[size_filter]
    class_names = class_names[size_filter]

    class_ids = np.arange(len(class_names), dtype=int)
    detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)  # (n, 4)  # (n,)
    detections = detections.with_nms(threshold=0.01)
    detections.class_id[detections.confidence < 0.05] = 902

    box_annotator = sv.BoxAnnotator(color_palette_custom)
    annotate_detections = detections[np.argsort(detections.class_id)[::-1]]
    annotated_frame = box_annotator.annotate(scene=undistorted_image.copy(), detections=annotate_detections)

    label_annotator = sv.LabelAnnotator(color_palette_custom)
    labels = [
        f"{class_id} {confidence:.2f}"
        for class_id, confidence in zip(annotate_detections.class_id, annotate_detections.confidence)
    ]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=annotate_detections, labels=labels)
    debug_images["pot_boxes"] = Image.fromarray(annotated_frame)


def warp(image: np.ndarray, tray: Tray):
    src_points = np.array(
        [tray.rect.top_left, tray.rect.top_right, tray.rect.bottom_right, tray.rect.bottom_left],
        dtype=np.float32,
    )
    width = tray.n_wide * POT_WIDTH
    height = tray.n_tall * POT_HEIGHT
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    return warped_image
