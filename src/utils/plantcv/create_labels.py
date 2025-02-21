# Create labels base on clean mask and optionally, multiple ROIs
import logging
import multiprocessing
import os

import numpy as np
from plantcv.plantcv import Objects, params
from plantcv.plantcv._debug import _debug
from skimage.color import label2rgb

from src.utils.plantcv.roi.quick_filter import quick_filter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_label(mask: np.ndarray, i: int, roi: Objects):
        filtered_mask = quick_filter(mask, roi)
        return (i + 1) * filtered_mask


def fast_create_labels(mask: np.ndarray, rois: Objects):
    """Create a labeled mask where connected regions of non-zero
    pixels are assigned a label value based on the provided
    region of interest (ROI).

    Inputs:
    mask            = mask image
    rois            = list of multiple ROIs (from roi.multi or roi.auto_grid)

    Returns:
    mask            = Labeled mask
    num_labels      = Number of labeled objects

    :param mask: numpy.ndarray
    :param rois: plantcv.plantcv.classes.Objects
    :return labeled_mask: numpy.ndarray
    :return num_labels: int
    """
    # Store debug mode
    debug = params.debug
    params.debug = None

    labeled_mask = np.zeros(mask.shape[:2], dtype=np.int32)
    num_labels = len(rois.contours)

    # with multiprocessing.Pool() as pool:
    #     results = pool.starmap(create_label, [(mask, i, roi) for i, roi in enumerate(rois)])

    # for result in results:
    #     labeled_mask += result

    for i, roi in enumerate(rois.contours):
        filtered_mask = quick_filter(mask, roi)
        labeled_mask += (i + 1) * filtered_mask

    # Restore debug parameter
    params.debug = debug
    colorful = label2rgb(labeled_mask)
    colorful2 = (255 * colorful).astype(np.uint8)

    _debug(colorful2, filename=os.path.join(params.debug_outdir, str(params.device) + "_label_colored_mask.png"))

    return labeled_mask, num_labels
