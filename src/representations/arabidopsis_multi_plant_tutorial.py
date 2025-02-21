# %% [markdown]
# # Arabidopsis Multi-Plant Tutorial
#
# This is a fully-functioning workflow that demonstrates how to analyze the shape, size, and color of individual arabidopsis plants grown in a tray. Similar methods will work for other plant species imaged in this layout until the plants grow large enough to obscure each other.
#

# %% [markdown]
# # Section 1: Importing Image and Libraries

# %%
# Set the notebook display method
# %matplotlib notebook

# %%
# Import libraries
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import matplotlib.pyplot as plt


# %% [markdown]
# ## Input/Ouput variables
#
# Using this while developing a workflow in Jupyter makes it easier to convert the workflow to a script later.

# %%
# Input/output options
args = WorkflowInputs(
    images=["../../data/z3c1--2022-12-31--01-00-01.png", "../../data/z3c1--2022-12-31--09-30-01.png"],
    names="image1,image2",
    result="arabidopsis_results.json",
    outdir=".",
    writeimg=True,
    debug="plot",
    sample_label="genotype"
    )


# %%
# Set debug to the global parameter
pcv.params.debug = args.debug

# Set plotting size (default = 100)
pcv.params.dpi = 100

# Increase text size and thickness to make labels clearer
# (size may need to be altered based on original image size)
pcv.params.text_size = 10
pcv.params.text_thickness = 20

# %% [markdown]
# ## Read the input image

# %%
# Inputs:
#   filename = Image file to be read in
#   mode     = How to read in the image; either 'native' (default),
#              'rgb', 'gray', 'csv', or 'envi'
img, path, filename = pcv.readimage(filename=args.image2)

# use homography to correct for lens distortion
import cv2
import numpy as np
corners = [[0, 0], [0, 1], [1, 1], [1, 0]]

# "top_left": {"x": 120, "y": 407},
# "top_right": {"x": 1676, "y": 282},
# "bottom_left": {"x": 200, "y": 1650},
# "bottom_right": {"x": 1772, "y": 1518},
#%%
src = np.array([[120, 407], [1676, 282], [200, 1650], [1772, 1518]], dtype=np.float32)
dst = np.array([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]], dtype=np.float32)

h, status = cv2.findHomography(src, dst)
img = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
# %% [markdown]
# # Section 2: Segmenting plant from background and identifying plant object(s)
#

# %% [markdown]
# ## Visualize colorspaces
#
# The visualization tool converts the color image into HSV and LAB colorspaces
# and displays the grayscale channels in a matrix so that they can be
# visualized simultaneously. The idea is to select a channel that maximizes
# the difference between the plant and the background pixels.

# %%
# Inputs:
#   rbg_img = original image
#   original_img = whether to includ the RGB image in the display:
#                  True (default) or False
colorspaces = pcv.visualize.colorspaces(rgb_img=img, original_img=False)
plt.imshow(colorspaces)
plt.show()
# %% [markdown]
# ## Convert the color image to grayscale
#
# Converts the input color image into the LAB colorspace
# and returns the A (green-magenta) channel as a grayscale
# image.

# %%
# Inputs:
#   rbg_img = original image
#   channel = desired colorspace ('l', 'a', or 'b')
a = pcv.rgb2gray_lab(rgb_img=img, channel='a')

# %% [markdown]
# ## Visualize the distribution of grayscale values
#
# A histogram can be used to visualize the distribution of values
# in an image. The histogram can aid in the selection of a
# threshold value. This is NOT helpful in parallel, only while building a workflow.
#
# For this image, the large peak between 100-140 are from the
# brighter background pixels. The smaller peak between 80-90
# are the darker plant pixels.

# %%
hist = pcv.visualize.histogram(img=a, bins=25)
hist

# %% [markdown]
# ## Threshold the grayscale image
#
# Use a threshold function (binary in this case) to segment the grayscale
# image into plant (white) and background (black) pixels. Using the
# histogram above, a threshold point between 90-110 will segment the
# plant and background peaks. Because the plants are the darker pixels
# in this image, use `object_type="dark"` to do an inverse threshold.

# %%
a_thresh = pcv.threshold.binary(gray_img=a, threshold=125, object_type='dark')
plt.imshow(a_thresh)
plt.show()
# %% [markdown]
# ## Remove small background noise
#
# Thresholding mostly labeled plant pixels white but also labeled
# small regions of the background white. The fill function removes
# "salt" noise from the background by filtering white regions by size.

# %%
a_fill = pcv.fill(bin_img=a_thresh, size=500)
plt.imshow(a_fill)
plt.show()

# %% [markdown]
# # Section 3: Define a region of interest for each plant
#
# Use the automatic grid detection tool to define a region of interest (ROI) for each pot
# in the tray. Each ROI will be associated with a plant later. The ROIs
# do not need to completely contain a whole plant but must only overlap a
# single plant each.

# %%
rois = pcv.roi.auto_grid(mask=a_fill, nrows=6, ncols=8, img=img)
#%%
# %% [markdown]
# # Section 4: Create a labeled mask
# In order to measure each plant separately, rather than as one object of disconnected blobs, we must create a labeled masked where each plant has a specific pixel value even in the case of disconnected leaves in the binary mask.
#
#

# %%
# Create a labeled mask, this function works very similarly to the roi.filter step above
# import time
# start_time = time.time()
# old_labeled_mask, num_plants = pcv.create_labels(mask=a_fill, rois=rois, roi_type="partial")
# # print(f"Time: {time.time() - start_time:.2f}")
# plt.imshow(old_labeled_mask)
# plt.show()

# %%
# V2 Create a labeled mask, this function works very similarly to the roi.filter step above
import time
import multiprocessing
from plantcv.plantcv._helpers import _roi_filter, _cv2_findcontours

"""PlantCV quick_filter module."""
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
    # summed = summed.round() #.astype("uint8")
    summed = summed.astype("int32")

    # Make sure the mask is binary
    summed[np.where(summed > 0)] = 1

    # Print/plot debug image
    params.debug = debug
    _debug(visual=summed, filename=os.path.join(params.debug_outdir, f"{params.device}_roi_filter.png"), cmap="gray")

    return summed

def process_roi(i, roi):
    filtered_mask = quick_filter(a_fill, roi)
    return (i + 1) * filtered_mask

def create_labels(mask, rois):
    # labeled_mask, num_plants = pcv.create_labels(mask=a_fill, rois=rois, roi_type="partial")
    # contours, hierarchy = _cv2_findcontours(a_fill)
    labeled_mask = np.zeros(a_fill.shape[:2], dtype=np.int32)
    # num_labels = len(rois.contours)


    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_roi, enumerate(rois))

    for result in results:
        labeled_mask += result

start_time = time.time()
create_labels(a_fill, rois)
print(f"Time: {time.time() - start_time:.2f}")
#%%
plt.imshow(labeled_mask)
plt.show()


#%%
print(old_labeled_mask.shape, old_labeled_mask.dtype, old_labeled_mask.max())
print(labeled_mask.shape, labeled_mask.dtype, labeled_mask.max())
# %% [markdown]
# # Section 5: Measure each plant


# %%
shape_img = pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=48)
# plt.imshow(shape_img)
# plt.show()
# %%
shape_img = pcv.analyze.color(rgb_img=img, labeled_mask=labeled_mask, n_labels=20, colorspaces="HSV")
# plt.imshow(shape_img)
# plt.show()
# %% [markdown]
# ## Save the results
#
# During analysis, measurements are stored in the background in the `outputs` class.
#
# This example includes image analysis for 'area', 'convex_hull_area', 'solidity', 'perimeter', 'width', 'height', 'longest_path', 'center_of_mass, 'convex_hull_vertices', 'object_in_frame', 'ellipse_center', 'ellipse_major_axis', 'ellipse_minor_axis', 'ellipse_angle', 'ellipse_eccentricity' while using `pcv.analyze.size`. Plus we have color trait information also!
#

# %%
pcv.outputs.save_results(filename=args.result, outformat="json")

# %%
