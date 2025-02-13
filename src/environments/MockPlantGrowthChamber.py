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


class MockPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self):
        super().__init__(None, None)
        self.reference_spectrum = np.array([0.199, 0.381, 0.162, 0.000, 0.166, 0.303])
        self.data_path = Path("data")
        self.data_iter = self.data_iterator()
        radius = 68
        parallelograms = [
            {
                "top_left": {
                    "x": 120,
                    "y": 407
                },
                "top_right": {
                    "x": 1676,
                    "y": 282
                },
                "bottom_left": {
                    "x": 170,
                    "y": 986
                },
                "bottom_right": {
                    "x": 1734,
                    "y": 855
                },
                "num_pots_wide": 8,
                "num_pots_tall": 3,
                "genotype": "WT"
            },
            {
                "top_left": {
                    "x": 162,
                    "y": 1060
                },
                "top_right": {
                    "x": 1726,
                    "y": 930
                },
                "bottom_left": {
                    "x": 200,
                    "y": 1650
                },
                "bottom_right": {
                    "x": 1772,
                    "y": 1518
                },
                "num_pots_wide": 8,
                "num_pots_tall": 3,
                "genotype": "WT"
            }
        ]
        self.plant_rois_list = []
        for parallelogram in parallelograms:
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
                    self.plant_rois_list.append({
                    "cx": pot[0],
                    "cy": pot[1],
                    "r": radius,
                    "number": len(self.plant_rois_list),
                    "genotype": parallelogram["genotype"],
                })
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

    def get_observation(self):
        self.time = time.time()
        timestamp = datetime.fromtimestamp(self.time)
        self.get_image()
        rows, self.shape_image = self.run_plant_cv_pipeline(self.image)
        rows = np.array(rows)
        print(rows.shape)
        observation = np.array(self.image)
        observation = np.ones(1)
        # how is this encoded?
        observation[0] = self.time
        return observation

    def data_iterator(self):
        for file in self.data_path.rglob("*.png"):
            print(file)
            image = Image.open(file)
            yield image

    def run_plant_cv_pipeline(self, image: Image):
        pcv.params.sample_label = "plant"
        pcv.outputs.clear()

        image_array = np.array(image)

        if self.experiment.undistortion is None:
            undistorted_image = image_array
        else:
            undistorted_image = self.experiment.undistortion.undistort(image_array)

        plant_mask = self.experiment.plant_hsv_threshold.threshold(undistorted_image)
        rois = self.experiment.plant_rois.create_pcv_rois(undistorted_image)
        labeled_mask, num_plants = pcv.create_labels(mask=plant_mask, rois=rois, roi_type="partial")
        shape_image = pcv.analyze.size(img=undistorted_image, labeled_mask=labeled_mask, n_labels=num_plants)

        # all of this is just for visualization
        for roi in self.plant_rois_list:
            cx, cy, r = roi["cx"], roi["cy"], roi["r"]
            shape_image = cv2.circle(shape_image, (int(cx), int(cy)), r, (0, 255, 255), 2)
        cyan = np.zeros_like(undistorted_image)
        cyan[:, :, 1] = 255
        cyan[:, :, 2] = 255
        alpha = 0.5
        mask_indices = plant_mask > 0
        shape_image[mask_indices] = shape_image[mask_indices] * (1 - alpha) + cyan[mask_indices] * alpha
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
