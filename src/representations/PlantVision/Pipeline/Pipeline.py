import csv
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
from csv_x_over_time import csv_x_over_time
from Experiment import Experiment
from plantcv import plantcv as pcv


def listener(outfilepath: Path, queue):
    with open(outfilepath, "a", newline="") as f:
        fieldnames = [
            "sample",
            "image_name",
            "timestamp",
            "date",
            "time",
            "plant_id",
            "genotype",
            "area",
            "in_bounds",
            "object_in_frame",
            "perimeter",
            "height",
            "width",
            "solidity",
            "center_of_mass_x",
            "center_of_mass_y",
            "ellipse_center_x",
            "ellipse_center_y",
            "ellipse_major_axis",
            "ellipse_minor_axis",
            "ellipse_angle",
            "ellipse_eccentricity",
            "convex_hull_vertices",
            "convex_hull_area",
            "longest_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.path.getsize(outfilepath) == 0:
            # write header
            writer.writeheader()

        while True:
            m = queue.get()
            if m == "kill":
                break
            writer.writerow(m)
            f.flush()


def process_single_image(experiment: Experiment, path: Path, queue, outimdir: Path):
    pcv.params.debug = None
    pcv.params.sample_label = "plant"
    image_path = str(path.absolute())
    image_name: str = path.name
    image_name_no_extension = path.stem
    timestamp = experiment.regex.getdate(image_name_no_extension)

    img = cv2.imread(image_path)

    pcv.outputs.clear()

    if img is None:
        return

    # undistort (optional)
    if experiment.undistortion is None:
        undistorted_img = img
    else:
        undistorted_img = experiment.undistortion.undistort(img)

    # mask and fill
    plant_mask = experiment.plant_hsv_threshold.threshold(undistorted_img)

    # define rois
    rois = experiment.plant_rois.create_pcv_rois(undistorted_img)

    # labeled mask
    labeled_mask, num_plants = pcv.create_labels(
        mask=plant_mask, rois=rois, roi_type="partial"
    )

    # analyze each plant
    shape_img = pcv.analyze.size(
        img=undistorted_img, labeled_mask=labeled_mask, n_labels=num_plants
    )

    outimagepath = str(outimdir / f"{image_name_no_extension}_processsed.png")
    cv2.imwrite(outimagepath, shape_img)

    observations = pcv.outputs.observations

    for sample in observations:
        # each sample represents one roi
        row = {}
        sample: str
        plant_num = int(sample.removeprefix("plant_"))
        label = f"{image_name_no_extension}_plant_{plant_num:02}"
        row["timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M")
        row["date"] = str(timestamp.date())
        row["time"] = str(timestamp.time())
        row["plant_id"] = plant_num
        row["sample"] = label
        row["image_name"] = image_name
        row["genotype"] = experiment.plant_rois.data[plant_num - 1].genotype

        for variable in observations[sample]:
            # special cases for tuples
            if variable == "center_of_mass":
                row["center_of_mass_x"] = observations[sample][variable]["value"][0]
                row["center_of_mass_y"] = observations[sample][variable]["value"][1]
            elif variable == "ellipse_center":
                row["ellipse_center_x"] = observations[sample][variable]["value"][0]
                row["ellipse_center_y"] = observations[sample][variable]["value"][1]
            # general case
            else:
                row[variable] = observations[sample][variable]["value"]

        queue.put(row)


class Pipeline:
    def __init__(self, experiment: Experiment, experiment_dir_path: Path) -> None:
        self.e: Experiment = experiment
        self.experiment_dir_path: Path = experiment_dir_path
        print(f"Validating experiment {experiment.ID}")

    def run(self):
        print(f"Running experiment {self.e.name}")
        resultsdirpath = self.experiment_dir_path / self.e.ID / "results"
        resultsdirpath.mkdir(exist_ok=True)

        # get images that are in the timeframe
        image_paths = [
            p for p in self.e.path.glob("*") if p.suffix.lower() in [".jpg", ".png"]
        ]
        print(f"Found {len(image_paths)} images")
        # print(image_paths)

        print("Checking which ones are in the timeframe")

        valid_paths = [
            p
            for p in image_paths
            if self.e.timeframe_list.time_in_list(self.e.regex.getdate(p.name))
        ]
        print(f"Found {len(valid_paths)} valid images in the timeframe")
        # print(valid_paths)

        now = datetime.now()
        daystr = now.strftime("%Y-%m-%d--%H-%M-%S")

        rundirpath = resultsdirpath / daystr
        rundirpath.mkdir(exist_ok=True)

        outpath = rundirpath / f"{self.e.name}.csv"

        outimdir = rundirpath / "images"
        outimdir.mkdir(exist_ok=True)

        manager = mp.Manager()

        q = manager.Queue()

        file_pool = mp.Pool(1)
        file_pool.apply_async(listener, (str(outpath), q))

        items = []
        for valid_image_path in valid_paths:
            items.append((self.e, valid_image_path, q, outimdir))

        print("PROCESSING...")
        time_start = time.time()

        with mp.Pool() as pool:
            pool.starmap(process_single_image, items)
        q.put("kill")

        file_pool.close()
        file_pool.join()

        print("Program finished!", flush=True)
        time_end = time.time()
        print("Time elapsed: ", time_end - time_start, flush=True)

        print("Creating AREA over time csv...")
        csv_x_over_time(str(outpath), "area")

        print("Done")
