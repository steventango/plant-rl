"""PlantCV quick_filter module.
https://github.com/danforthcenter/plantcv/blob/main/plantcv/plantcv/roi/quick_filter.py
"""
import os
import cv2
import numpy as np
from skimage.measure import label
from skimage.color import label2rgb
from plantcv.plantcv import params
from plantcv.plantcv.roi import roi2mask
from plantcv.plantcv._debug import _debug
from plantcv.plantcv.logical_and import logical_and


def quick_filter(mask, roi):
    """Quickly filter a binary mask using a region of interest.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask to filter.
    roi : plantcv.plantcv.classes.Objects
        PlantCV ROI object.

    Returns
    -------
    numpy.ndarray
        Filtered binary mask.
    """
    # Increment the device counter
    params.device += 1

    # Store debug
    debug = params.debug
    params.debug = None

    # Label objects in the image from 1 to n (labeled mask)
    labels, num = label(label_image=mask, return_num=True)

    # Convert the input ROI to a binary mask (only works on single ROIs)
    roi_mask = roi2mask(img=mask, roi=roi)

    # Convert the labeled mask and ROI mask to float data types
    roi_mask = roi_mask.astype(float)
    labels = labels.astype(float)

    # Set the ROI mask value to 0.5
    roi_mask[np.where(roi_mask == 255)] = 0.5

    # Add the labeled mask and ROI mask together
    summed = roi_mask + labels

    # For each label, if at least one pixel of the object overlaps the ROI
    # set all the label values to the label plus 0.5
    for i in range(1, num + 1):
        if i + 0.5 in summed:
            summed[np.where(summed == i)] = i + 0.5

    # Objects that do not overlap the ROI will round to an integer and have
    # the same value before and after rounding.
    # Objecs that overlap the ROI will round up/down and will not have the same value
    # Where the values are equal (not overlapping)
    summed[np.where(summed == summed.round())] = 0

    # The summed image now only contains objects that overlap the ROI
    # Subtract 0.5 to remove the ROI mask
    summed = summed - 0.5

    # Round and set the data type back to uint8
    summed = summed.round().astype("uint8")

    # Make sure the mask is binary
    summed[np.where(summed > 0)] = 255

    # Print/plot debug image
    params.debug = debug
    _debug(visual=summed, filename=os.path.join(params.debug_outdir, f"{params.device}_roi_filter.png"), cmap="gray")

    return summed
