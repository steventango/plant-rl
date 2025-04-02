from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from plantcv import plantcv as pcv
from skimage.exposure import match_histograms, equalize_adapthist

from .zones import POT_HEIGHT, POT_WIDTH, SCALE, Tray

reference = np.array(Image.open(Path(__file__).parent / "reference.png"))
reference = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)


def process_image(image: np.ndarray, trays: list[Tray], debug_images: dict[str, Image]):
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

    lab = cv2.cvtColor(warped_image, cv2.COLOR_RGB2LAB)
    matched = match_histograms(lab, reference, channel_axis=-1)
    matched = cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    debug_images["matched"].append(matched)

    colorspaces = pcv.visualize.colorspaces(rgb_img=matched, original_img=False)
    debug_images["colorspaces"].append(colorspaces)

    gray_image = pcv.rgb2gray_lab(rgb_img=matched, channel="a")
    debug_images["gray"].append(gray_image)

    gray_image = gray_image.astype(np.uint8)
    equalized = equalize_adapthist(gray_image, kernel_size=pot_width // 3, clip_limit=0.03)
    equalized = (equalized * 255).astype(np.uint8)
    debug_images["equalized"].append(equalized)
    OFFSET = 50
    mask = pcv.threshold.mean(gray_img=equalized, ksize=1 * pot_width, offset=OFFSET, object_type="dark")
    debug_images["mask"].append(mask)
    FILL_THRESHOLD = 0.001 * pot_width**2
    mask = pcv.fill(mask, FILL_THRESHOLD)
    debug_images["mask_filled"].append(mask)
    stats = []
    r = int(POT_WIDTH / 3)
    coords = []
    for j in range(tray.n_tall):
        for i in range(tray.n_wide):
            y = j * pot_width + pot_width // 2
            x = i * pot_width + pot_width // 2
            coords.append((x, y))

    roi = pcv.roi.multi(warped_image, coord=coords, radius=r)
    labeled_mask, num_plants = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")
    from plantcv.plantcv import params
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
