from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from PIL import Image
from plantcv import plantcv as pcv
from plantcv.plantcv import params
from skimage.exposure import equalize_adapthist, match_histograms
from supervision.draw.color import ColorPalette

from utils.grounded_sam2 import GroundingDino, SAM2

from .zones import POT_HEIGHT, POT_WIDTH, SCALE, Tray

grounding_dino = GroundingDino()
sam2 = SAM2()

def process_image(image: np.ndarray, trays: list[Tray], debug_images: dict[str, Image.Image]):
    if not trays:
        raise ValueError("No trays provided")
    all_plant_stats = []
    debug_tray_images = defaultdict(list)

    camera_matrix = np.array([[1800.0, 0.0, 1296.0], [0.0, 1800.0, 972.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
    # convert image from RGBA to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    debug_images["undistorted"] = Image.fromarray(undistorted_image)
    for tray in trays:
        plant_stats = process_tray(undistorted_image, tray, debug_tray_images)
        all_plant_stats.extend(plant_stats)
    # convert debug_images to PIL images
    for key, images in debug_tray_images.items():
        images = np.array(images)
        images = images.reshape(len(trays), *images.shape[1:])
        # stack images on the longest axis
        if images.shape[1] > images.shape[2]:
            debug_images[key] = Image.fromarray(np.hstack(images))
        else:
            debug_images[key] = Image.fromarray(np.vstack(images))
    # convert all_plant_stats to pandas dataframe
    df = pd.DataFrame(all_plant_stats)
    df.plant_id = df.index
    return df


def process_tray(image: np.ndarray, tray: Tray, debug_images: dict[str, list[np.ndarray]]):
    src_points = np.array(
        [tray.rect.top_left, tray.rect.top_right, tray.rect.bottom_right, tray.rect.bottom_left],
        dtype=np.float32,
    )
    pot_width = POT_WIDTH
    width = tray.n_wide * POT_WIDTH
    height = tray.n_tall * POT_HEIGHT
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    debug_images["warped"].append(warped_image)

    pil_image = Image.fromarray(warped_image)

    lab = cv2.cvtColor(warped_image, cv2.COLOR_RGB2LAB)
    gray_image = pcv.rgb2gray_lab(rgb_img=lab, channel="a")
    debug_images["gray"].append(gray_image)

    gray_image = gray_image.astype(np.uint8)
    equalized = equalize_adapthist(gray_image, kernel_size=pot_width // 3, clip_limit=0.1)
    equalized = (equalized * 255).astype(np.uint8)

    boxes, confidences, class_names = grounding_dino.inference(
        image=pil_image,
        text_prompt="plant.",
        box_threshold=0.05,
        text_threshold=0.05,
    )

    # filter out boxes that are bigger than 2 times the pot size
    size_filter = (boxes[:, 2] - boxes[:, 0] < pot_width * 2) & (boxes[:, 3] - boxes[:, 1] < pot_width * 2)

    boxes = boxes[size_filter]
    confidences = confidences[size_filter]
    class_names = class_names[size_filter]

    coords = []
    sigma = pot_width
    for j in range(tray.n_tall):
        for i in range(tray.n_wide):
            y = j * pot_width + pot_width // 2
            x = i * pot_width + pot_width // 2
            coords.append((x, y))

            # box is in range if box center is inside the pot
            box_centers = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            boxes_in_range = (
                (box_centers[0] > x - pot_width // 2) & (box_centers[0] < x + pot_width // 2) &
                (box_centers[1] > y - pot_width // 2) & (box_centers[1] < y + pot_width // 2)
            )

            # RBF score for boxes closer to the center of the pot
            score = np.exp(-((x - boxes[boxes_in_range, 0]) ** 2 + (y - boxes[boxes_in_range, 1]) ** 2) / (sigma / 2) ** 2)
            confidences[boxes_in_range] *= score

            # only keep the box with the highest confidence
            if boxes_in_range.nonzero()[0].size < 2:
                continue

            max_confidence = confidences[boxes_in_range].max()
            for index in boxes_in_range.nonzero()[0]:
                if confidences[index] < max_confidence:
                    confidences[index] = 0

    class_name_id_map = {name: i + 1 for i, name in enumerate(np.unique(class_names))}
    class_ids = np.array([class_name_id_map[name] for name in class_names])

    detections = sv.Detections(
        xyxy=boxes,  # (n, 4)
        confidence=confidences,  # (n,)
        class_id=class_ids
    )
    detections = detections.with_nms(threshold=0.01)
    detections = detections[detections.confidence > 0.03]

    if not len(detections.xyxy):
        # TODO: better handling of no boxes
        return [{
            "area": 0,
        }]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=warped_image.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    labels = [
        f"{confidence:.2f}" for confidence in detections.confidence
    ]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    debug_images["boxes"].append(annotated_frame)

    masks, *_ = sam2.inference(
        image=pil_image,
        boxes=detections.xyxy,
    )

    detections.mask = masks.astype(bool)

    # filter out masks with area > 0.6 * pot_width * pot_width
    mask_areas = np.sum(detections.mask, axis=(1, 2))
    size_filter = mask_areas < 0.6 * pot_width * pot_width
    detections = detections[size_filter]

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=warped_image.copy(), detections=detections)
    debug_images["masks"].append(annotated_frame)

    sam_mask = detections.mask.any(axis=0)
    sam_mask = (sam_mask * 255).astype(np.uint8)
    debug_images["sam_mask"].append(sam_mask)

    debug_images["equalized"].append(equalized)
    equalized = cv2.bitwise_and(equalized, equalized, mask=sam_mask)
    debug_images["equalized_post_and"].append(equalized)
    mask = pcv.threshold.binary(equalized, threshold=127, object_type="light")
    debug_images["mask"].append(mask)

    stats = []
    r = int(POT_WIDTH / 3)

    roi = pcv.roi.multi(warped_image, coord=coords, radius=r)
    labeled_mask, num_plants = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")

    params.line_thickness = 1

    shape_image = pcv.analyze.size(img=warped_image, labeled_mask=labeled_mask, n_labels=num_plants)
    for x, y in coords:
        shape_image = cv2.circle(shape_image, (x, y), r, (0, 255, 255), 1)

    stats = []
    for sample, variables in pcv.outputs.observations.items():
        row = {}
        plant_num = int(sample.removeprefix("default_"))
        row["plant_id"] = plant_num

        for variable, value in variables.items():
            if variable == "center_of_mass":
                row["center_of_mass_x"], row["center_of_mass_y"] = value["value"]
            elif variable == "ellipse_center":
                row["ellipse_center_x"], row["ellipse_center_y"] = value["value"]
            else:
                row[variable] = value["value"]
        row["area"] /= SCALE**2
        stats.append(row)


    for row, coord in zip(stats, coords):
        x, y = coord
        area = row["area"]
        if area is not None:
            text = f"{area:.2f} mm^2"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x - text_size[0] // 2
            text_y = y + int(r * 1.25) + text_size[1] // 2
            cv2.putText(
                shape_image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    debug_images["shape_image"].append(shape_image)

    return stats
