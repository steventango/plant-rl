import uuid
from pathlib import Path
from Regex import Regex
from Undistortion import Undistortion
from Roi import RoiList
from HSVThreshold import HSVThreshold
from TimeFrame import TimeFrameList
import json


class Experiment:
    def __init__(
        self,
        timeframe_list: TimeFrameList,
        name: str,
        description: str,
        path: Path,
        regex: Regex,
        undistortion: Undistortion | None,
        plant_hsv_threshold: HSVThreshold,
        plant_rois: RoiList,
        ID: str = None,
    ) -> None:
        self.timeframe_list: TimeFrameList = timeframe_list
        self.name: str = name
        self.description: str = description
        self.path: Path = path
        self.regex: Regex = regex
        self.undistortion: Undistortion | None = undistortion
        self.plant_hsv_threshold: HSVThreshold = plant_hsv_threshold
        self.plant_rois: RoiList = plant_rois

        if ID is None:
            self.ID: str = str(uuid.uuid1())
        else:
            self.ID = ID

    def save(self, dirpath: Path):
        self.timeframe_list.save_to_file(dirpath)

        if self.undistortion is not None:
            self.undistortion.save_to_file(dirpath)

        self.plant_rois.save_to_file(dirpath)
        self.plant_hsv_threshold.save_to_file(dirpath)

        # make metadata file and save that
        metadata_json = {
            "title": self.name,
            "description": self.description,
            "path": str(self.path),
            "filename_format": self.regex.name,
        }

        with open(dirpath / (self.ID + ".json"), "w") as f:
            json.dump(metadata_json, f)
