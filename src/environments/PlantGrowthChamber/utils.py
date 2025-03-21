from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from plantcv import plantcv as pcv

from .zones import POT_HEIGHT, POT_WIDTH, SCALE, Tray


def process_image(image: np.ndarray, trays: list[Tray], debug_images: dict[str, Image]):
    if not trays:
        raise ValueError("No trays provided")
    all_plant_stats = []
    debug_tray_images = defaultdict(list)

    camera_matrix = np.array([[1800.0, 0.0, 1296.0], [0.0, 1800.0, 972.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    debug_images["undistorted"] = Image.fromarray(undistorted_image)
    for tray in trays:
        plant_stats = process_tray(undistorted_image, tray, debug_tray_images)
        all_plant_stats.extend(plant_stats)
    # convert debug_images to PIL images
    for key, images in debug_tray_images.items():
        images = np.array(images)
        images = images.reshape(len(trays), *images.shape[1:])
        debug_images[key] = Image.fromarray(np.vstack(images))
    # convert all_plant_stats to pandas dataframe
    df = pd.DataFrame([stat for sublist in all_plant_stats for stat in sublist])
    df.plant_id = df.index
    return df


def process_tray(image: np.ndarray, tray: Tray, debug_images: dict[str, list[np.ndarray]]):
    src_points = np.array(
        [tray.rect.top_left, tray.rect.top_right, tray.rect.bottom_right, tray.rect.bottom_left],
        dtype=np.float32,
    )
    width = tray.n_wide * POT_WIDTH
    height = tray.n_tall * POT_HEIGHT
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    debug_images["warped"].append(warped_image)

    MARGIN = 0.7
    without_border_images = []
    for i in range(tray.n_wide):
        for j in range(tray.n_tall):
            without_border_image = get_pot_crop(warped_image, i, j, MARGIN, POT_WIDTH)
            without_border_images.append(without_border_image)
    # combine without_border_images into one image (n_wide x n_tall)
    without_border_images = np.array(without_border_images)
    without_border_images = without_border_images.reshape(tray.n_tall, tray.n_wide, *without_border_images.shape[1:])
    # reassemble images into one image
    without_border_image = np.vstack([np.hstack(row) for row in without_border_images])
    debug_images["without_border"].append(without_border_image)

    colorspaces = pcv.visualize.colorspaces(rgb_img=without_border_image, original_img=False)
    debug_images["colorspaces"].append(colorspaces)
    gray_image = pcv.rgb2gray_lab(rgb_img=without_border_image, channel="a")
    debug_images["gray"].append(gray_image)
    normalized_gray_image = (gray_image - np.mean(gray_image)) / np.std(gray_image)
    normalized_gray_image = 127 + 64 * normalized_gray_image
    normalized_gray_image = np.clip(normalized_gray_image, 0, 255).astype(np.uint8)
    debug_images["normalized_gray"].append(normalized_gray_image)
    pot_width = int(POT_WIDTH * MARGIN)
    mask = pcv.threshold.mean(gray_img=normalized_gray_image, ksize=3 * pot_width, offset=1.5 * 64, object_type="dark")
    debug_images["mask"].append(mask)
    FILL_THRESHOLD = 0.005 * pot_width**2
    mask = pcv.fill(mask, FILL_THRESHOLD)
    debug_images["mask_filled"].append(mask)
    debug_pot_images = defaultdict(list)
    stats = []
    for i in range(tray.n_wide):
        for j in range(tray.n_tall):
            pot_image = get_pot_crop(without_border_image, i, j, 1, pot_width)
            pot_mask = get_pot_crop(mask, i, j, 1, pot_width)
            shape_image, stat = process_plant(pot_image, pot_mask, debug_pot_images)
            stats.append(stat)
    # recombine debug_pot_images into one image (n_wide x n_tall)
    for key, images in debug_pot_images.items():
        images = np.array(images)
        images = images.reshape(tray.n_tall, tray.n_wide, *images.shape[1:])
        # reassemble images into one image
        debug_images[key].append(np.vstack([np.hstack(row) for row in images]))
    return stats


def get_pot_crop(image: np.ndarray, i: int, j: int, margin: float, pot_width):
    x = i * pot_width
    y = j * pot_width
    crop = image[y : y + pot_width, x : x + pot_width]
    # get the center of the crop with margin
    x = int(crop.shape[1] / 2)
    y = int(crop.shape[0] / 2)
    r = int(pot_width * margin) // 2
    crop2 = crop[y - r : y + r, x - r : x + r]
    return crop2


def process_plant(image: np.ndarray, mask, debug_images: dict[str, list[np.ndarray]]):
    x = int(image.shape[1] / 2)
    y = int(image.shape[0] / 2)
    r = POT_WIDTH // 4
    roi = pcv.roi.multi(image, coord=[(x, y)], radius=r)
    # plant_mask = plant_mask.astype(np.uint8) * 255
    labeled_mask, num_plants = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")
    from plantcv.plantcv import params
    params.line_thickness = 1
    # convert image from RGBA to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    shape_image = pcv.analyze.size(img=image, labeled_mask=labeled_mask, n_labels=num_plants)
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
        stats.append(row)

        row["area"] /= SCALE**2

    for row in stats:
        area = row["area"]
        if area is not None:
            cv2.putText(
                shape_image,
                f"{area:.2f} mm^2",
                (x - int(r * 1.1), y - int(r * 1.1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    debug_images["shape_image"].append(shape_image)
    return shape_image, stats
