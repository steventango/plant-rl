from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from cv2 import KMEANS_RANDOM_CENTERS, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, kmeans
from PIL import Image
from plantcv import plantcv as pcv

from .zones import POT_HEIGHT, POT_WIDTH, Tray


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
    colorspaces = pcv.visualize.colorspaces(rgb_img=warped_image, original_img=False)
    debug_images["colorspaces"].append(colorspaces)
    image_a = pcv.rgb2gray_lab(rgb_img=warped_image, channel='a')
    gray_image = image_a
    debug_images["gray"].append(gray_image)
    mask = pcv.threshold.mean(gray_img=gray_image, ksize=POT_WIDTH / 3, offset=5, object_type="dark")
    debug_images["mask"].append(mask)
    FILL_THRESHOLD = 0.0015 * POT_WIDTH * POT_HEIGHT
    mask = pcv.fill(mask, FILL_THRESHOLD)
    debug_images["mask_filled"].append(mask)
    cropping_positions_image = warped_image.copy()
    debug_pot_images = defaultdict(list)
    stats = []
    MARGIN = 0.8
    for i in range(tray.n_wide):
        for j in range(tray.n_tall):
            pot_image = get_pot_crop(warped_image, i, j, MARGIN)
            pot_mask = get_pot_crop(mask, i, j, MARGIN)
            shape_image, stat = process_plant(pot_image, pot_mask, debug_pot_images)
            stats.append(stat)
            # Visualize cropping positions
            x = i * POT_WIDTH
            y = j * POT_HEIGHT
            crop_width = int(POT_WIDTH)
            crop_height = int(POT_HEIGHT)
            x_offset = (POT_WIDTH - crop_width) // 2
            y_offset = (POT_HEIGHT - crop_height) // 2
            cv2.rectangle(cropping_positions_image, (x + x_offset, y + y_offset), (x + x_offset + crop_width, y + y_offset + crop_height), (0, 255, 0), 2)
    debug_images["cropping_positions"].append(cropping_positions_image)
    # recombine debug_pot_images into one image (n_wide x n_tall)
    for key, images in debug_pot_images.items():
        images = np.array(images)
        images = images.reshape(tray.n_tall, tray.n_wide, *images.shape[1:])
        # reassemble images into one image
        debug_images[key].append(np.vstack([np.hstack(row) for row in images]))
    return stats


def get_pot_crop(image: np.ndarray, i: int, j: int, margin: float):
    x = i * POT_WIDTH
    y = j * POT_HEIGHT
    crop = image[y : y + POT_HEIGHT, x  : x  + POT_WIDTH]
    # get the center of the crop with margin
    x = int(crop.shape[1] / 2)
    y = int(crop.shape[0] / 2)
    r = int(POT_WIDTH * margin) // 2
    crop2 = crop[y - r : y + r, x - r : x + r]
    return crop2


def process_plant(image: np.ndarray, mask, debug_images: dict[str, list[np.ndarray]]):
    x = int(image.shape[1] / 2)
    y = int(image.shape[0] / 2)
    r = POT_WIDTH // 3
    roi = pcv.roi.multi(image, coord=[(x, y)], radius=r)
    # plant_mask = plant_mask.astype(np.uint8) * 255
    labeled_mask, num_plants = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")
    from plantcv.plantcv import params
    params.line_thickness = 1
    # convert image from RGBA to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    shape_image = pcv.analyze.size(img=image, labeled_mask=labeled_mask, n_labels=num_plants)
    shape_image = cv2.circle(shape_image, (x, y), r, (0, 255, 255), 1)
    debug_images["shape_image"].append(shape_image)

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
    return shape_image, stats


# def get_plant_mask(gray_image: np.ndarray):

    # mask = pcv.threshold.otsu(gray_img=gray_image, object_type="dark")
    # mask = pcv.threshold.triangle(gray_img=gray_image, object_type="dark")
    # return mask
    # vector = image.reshape(-1, 3)s
    # compactness, labels, centers = kmeans(
    #     data=vector.astype(np.float32),
    #     K=2,
    #     bestLabels=None,
    #     criteria=(TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0),
    #     attempts=20,
    #     flags=KMEANS_RANDOM_CENTERS,
    # )

    # # Apply the RGB values of the cluster centers to all pixel labels
    # colours = centers[labels].reshape(-1, 3)
    # labels = labels.reshape(-1)

    # img_colors = colours.reshape(image.shape).astype(np.uint8)

    # # count labels
    # unique, counts = np.unique(labels, return_counts=True)
    # # get the label with the 2nd largest count
    # index = np.argsort(counts)[0]
    # plant_label = unique[index]
    # plant_mask = labels == plant_label
    # plant_mask = plant_mask.reshape(image.shape[:2])
    # return plant_mask, img_colors
