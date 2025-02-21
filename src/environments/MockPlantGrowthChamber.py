import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plantcv import plantcv as pcv

from environments.PlantGrowthChamber import PlantGrowthChamber
from representations.PlantVision.Pipeline.Experiment import Experiment
from representations.PlantVision.Pipeline.HSVThreshold import HSVThreshold
from representations.PlantVision.Pipeline.Roi import RoiList
from representations.PlantVision.Pipeline.Undistortion import Undistortion
from utils.plantcv.create_labels import fast_create_labels

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MockPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self):
        super().__init__(None, None)
        self.reference_spectrum = np.array([0.199, 0.381, 0.162, 0.000, 0.166, 0.303])
        self.data_path = Path("data")
        self.data_iter = self.data_iterator()
        radius = 68
        self.parallelograms = [
            {
                "top_left": {"x": 120, "y": 407},
                "top_right": {"x": 1676, "y": 282},
                "bottom_left": {"x": 170, "y": 986},
                "bottom_right": {"x": 1734, "y": 855},
                "num_pots_wide": 8,
                "num_pots_tall": 3,
                "genotype": "WT",
            },
            {
                "top_left": {"x": 162, "y": 1060},
                "top_right": {"x": 1726, "y": 930},
                "bottom_left": {"x": 200, "y": 1650},
                "bottom_right": {"x": 1772, "y": 1518},
                "num_pots_wide": 8,
                "num_pots_tall": 3,
                "genotype": "WT",
            },
        ]
        self.plant_rois_list = []
        for parallelogram in self.parallelograms:
            top_left = np.array([parallelogram["top_left"]["x"], parallelogram["top_left"]["y"]])
            top_right = np.array([parallelogram["top_right"]["x"], parallelogram["top_right"]["y"]])
            top_vector = top_right - top_left
            bottom_left = np.array([parallelogram["bottom_left"]["x"], parallelogram["bottom_left"]["y"]])
            bottom_right = np.array([parallelogram["bottom_right"]["x"], parallelogram["bottom_right"]["y"]])
            bottom_vector = bottom_right - bottom_left

            for i in range(parallelogram["num_pots_wide"]):
                for j in range(parallelogram["num_pots_tall"]):
                    pot_top = top_left + top_vector * (i + 0.5) / parallelogram["num_pots_wide"]
                    pot_bottom = bottom_left + bottom_vector * (i + 0.5) / parallelogram["num_pots_wide"]
                    vertical_vector = pot_bottom - pot_top
                    pot = pot_top + vertical_vector * (j + 0.5) / parallelogram["num_pots_tall"]
                    self.plant_rois_list.append(
                        {
                            "cx": pot[0],
                            "cy": pot[1],
                            "r": radius,
                            "number": len(self.plant_rois_list),
                            "genotype": parallelogram["genotype"],
                        }
                    )
        print("length of plant roi list:", len(self.plant_rois_list))
        plant_rois = RoiList()
        plant_rois.load_from_list(self.plant_rois_list)
        self.experiment = Experiment(
            timeframe_list=None,
            name="Plant Growth Chamber",
            description="Plant Growth Chamber",
            path=None,
            regex=None,
            undistortion=Undistortion(
                # **undistortion
                fx=1800.0,
                cx=1296.0,
                fy=1800.0,
                cy=972.0,
                k1=0.0,
                k2=0.0,
                k3=0.0,
                k4=0.0,
            ),
            plant_hsv_threshold=HSVThreshold(
                # **hsv_threshold
                hl=20,
                hh=80,
                sl=10,
                sh=255,
                vl=125,
                vh=255,
                fill=50,
                invert=False,
            ),
            plant_rois=plant_rois,
        )

    def get_image(self):
        try:
            self.image = next(self.data_iter)
        except StopIteration:
            self.data_iter = self.data_iterator()
            self.image = next(self.data_iter)

    def put_action(self, action):
        pass

    def close(self):
        pass

    # def get_observation(self):
    #     self.time = time.time()
    #     # timestamp = datetime.fromtimestamp(self.time)
    #     self.get_image()
    #     # rows, self.shape_image = self.run_plant_cv_pipeline(self.image)
    #     # observation = np.array(self.image)
    #     # observation = np.ones(1)
    #     # how is this encoded?
    #     observation[0] = (self.time)
    #     return observation

    def data_iterator(self):
        for file in self.data_path.rglob("*.png"):
            print(file)
            image = Image.open(file)
            yield image

    # TODO: move this out of the class
    def run_plant_cv_pipeline(self, image: Image):
        pcv.params.sample_label = "plant"
        pcv.outputs.clear()

        image_array = np.array(image)

        # if self.experiment.undistortion is None:
        #     undistorted_image = image_array
        # else:
        start_time = time.time()
        # undistorted_image = ?
        src = np.array([[120, 407], [1676, 282], [200, 1650], [1772, 1518]], dtype=np.float32)
        dst = np.array([[0, 0], [image_array.shape[1], 0], [0, image_array.shape[0]], [image_array.shape[1], image_array.shape[0]]], dtype=np.float32)
        h, status = cv2.findHomography(src, dst)
        undistorted_image = cv2.warpPerspective(image_array, h, (image_array.shape[1], image_array.shape[0]))
        end_time = time.time()
        logger.info(f"undistortion took {end_time - start_time} seconds")

        plant_mask = self.compute_plant_mask(undistorted_image)
        start_time = time.time()
        pcv.fill(plant_mask, sulf.experiment.plant_hsv_threshold.fill)
        end_time = time.time()
        logger.info(f"fill took {end_time - start_time} seconds")
        start_time = time.time()
        # rois = self.experiment.plant_rois.create_pcv_rois(undistorted_image)
        rois = pcv.roi.auto_grid(mask=plant_mask, nrows=6, ncols=8, img=undistorted_image)
        end_time = time.time()
        logger.info(f"create_pcv_rois took {end_time - start_time} seconds")
        start_time = time.time()
        labeled_mask, num_plants = fast_create_labels(mask=plant_mask, rois=rois)
        end_time = time.time()
        logger.info(f"create_labels took {end_time - start_time} seconds")
        start_time = time.time()
        shape_image = pcv.analyze.size(img=undistorted_image, labeled_mask=labeled_mask, n_labels=num_plants)
        end_time = time.time()
        logger.info(f"analyze.size took {end_time - start_time} seconds")

        # all of this is just for visualization
        for roi in self.plant_rois_list:
            cx, cy, r = roi["cx"], roi["cy"], roi["r"]
            shape_image = cv2.circle(shape_image, (int(cx), int(cy)), r, (0, 255, 255), 2)

        plant_mask = np.stack([plant_mask] * 3, axis=2)

        shape_image = np.concatenate(
            [plant_mask, shape_image], axis=1
        )
        shape_image = Image.fromarray(shape_image)

        rows = []
        for sample, variables in pcv.outputs.observations.items():
            row = {}
            plant_num = int(sample.removeprefix("plant_"))
            row["plant_id"] = plant_num

            for variable, value in variables.items():
                if variable == "center_of_mass":
                    row["center_of_mass_x"], row["center_of_mass_y"] = value["value"]
                elif variable == "ellipse_center":
                    row["ellipse_center_x"], row["ellipse_center_y"] = value["value"]
                else:
                    row[variable] = value["value"]
            rows.append(row)

        return rows, shape_image

    def compute_plant_mask(self, undistorted_image):
        start_time = time.time()
        plant_mask = np.zeros(undistorted_image.shape, dtype=np.uint8)
        plant_mask = plant_mask.sum(axis=2) > 0
        plant_mask = plant_mask.reshape(-1)
        img_data = undistorted_image.reshape(-1, 3)
        for roi in self.plant_rois_list:
            tray_mask = np.zeros(undistorted_image.shape, dtype=np.uint8)
            cx, cy, r = roi["cx"], roi["cy"], roi["r"]
            tray_mask = cv2.circle(tray_mask, (int(cx), int(cy)), int(1.5 * r), (255,), cv2.FILLED)
            tray_mask = tray_mask.sum(axis=2) > 0
            tray_mask = tray_mask.reshape(-1)
            tray_img_data = img_data[tray_mask]
            compactness, labels, centers = cv2.kmeans(
                data=tray_img_data.astype(np.float32),
                K=2,
                bestLabels=None,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 5, 1.0),
                attempts=1,
                flags=cv2.KMEANS_RANDOM_CENTERS,
            )

            # Apply the RGB values of the cluster centers to all pixel labels
            colours = centers[labels].reshape(-1, 3)
            labels = labels.reshape(-1)

            # Reshape array to the original image shape
            img_colours = undistorted_image.copy()
            img_colours = img_colours.reshape(-1, 3)
            img_colours[tray_mask] = colours
            img_colours = img_colours.reshape(undistorted_image.shape)
            # count labels
            unique, counts = np.unique(labels, return_counts=True)
            # smallest label is plant?
            plant_label = unique[np.argmin(counts)]  # assume smallest count corresponds to plant
            plant_mask[tray_mask] = labels == plant_label

        plant_mask = plant_mask.reshape(undistorted_image.shape[:2])
        # convert to cv::UMat
        plant_mask = plant_mask.astype(np.uint8) * 255
        end_time = time.time()
        logger.info(f"compute_plant_mask took {end_time - start_time} seconds")
        return plant_mask
