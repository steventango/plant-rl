import os
from pathlib import Path
import json
import shutil
from Regex import Regex, RegexBuilder
from Undistortion import (
    Undistortion,
    load_undistortion_from_dict,
    load_undistortion_from_file,
)
from Roi import RoiList
from HSVThreshold import (
    HSVThreshold,
    load_threshold_from_dict,
    load_threshold_from_file,
)
from TimeFrame import TimeFrameList
from Experiment import Experiment
from Pipeline import Pipeline


class ExperimentManager:
    def __init__(self, list_exps=False) -> None:
        self.experiment_dir_path = Path(Path.cwd() / "Experiments")

        self.experiments = self.load_all_exp_titles()
        if list_exps:
            print("Experiments loaded:")
            for e in self.experiments:
                print(e, "\n")

    def get_exp_metadata(self, expid: str):
        with open(Path(self.experiment_dir_path) / expid / (expid + ".json"), "r") as f:
            data = json.load(f)
            return {
                "title": data["title"],
                "description": data["description"],
                "filename_format": data["filename_format"],
            }

    # Creating / saving an experiment requires having all the required components
    def load_all_exp_titles(self):
        # return a list of tuples (name, id)
        l = []
        for expid in os.listdir(self.experiment_dir_path):
            if expid == ".DS_Store":
                continue
            meta = self.get_exp_metadata(expid)
            t = (expid, meta["title"], meta["description"])
            l.append(t)
        # print([x for x in os.walk(self.experiment_dir_path)])
        return l

    def load_experiment(self, expid: str):
        # given an experiment id, load the Experiment Object associated with it
        dirpath = self.experiment_dir_path / expid

        with open(dirpath / (expid + ".json"), "r") as f:
            metadata = json.load(f)

        # check if undistortion exists

        tfl = TimeFrameList()
        tfl.load_from_file(dirpath / "timeframe.json")

        if (dirpath / "undistort.json").is_file():
            undistortion = load_undistortion_from_file(dirpath / "undistort.json")
        else:
            undistortion = None

        threshold = load_threshold_from_file(dirpath / "threshold.json")
        rois = RoiList()
        rois.load_from_file(dirpath / "rois.json")

        return Experiment(
            timeframe_list=tfl,
            name=metadata["title"],
            description=metadata["description"],
            path=Path(metadata["path"]),
            regex=RegexBuilder(metadata["filename_format"]).get_regex(),
            undistortion=undistortion,
            plant_hsv_threshold=threshold,
            plant_rois=rois,
            ID=expid,
        )

    def create_experiment(
        self,
        name: str,
        timeframe_list: list,
        description: str,
        path: str,
        regex: str,
        undistortion: dict | None,
        threshold: dict,
        rois: list,
    ):
        if undistortion is not None:
            undistortion = load_undistortion_from_dict(undistortion)

        tfl = TimeFrameList()
        tfl.load_from_list(timeframe_list)

        rl = RoiList()
        rl.load_from_list(rois)

        experiment = Experiment(
            timeframe_list=tfl,
            name=name,
            description=description,
            path=Path(path),
            regex=RegexBuilder(regex).get_regex(),
            undistortion=undistortion,
            plant_hsv_threshold=load_threshold_from_dict(threshold),
            plant_rois=rl,
        )
        dirpath = self.experiment_dir_path / experiment.ID
        dirpath.mkdir()
        experiment.save(dirpath)
        return experiment.ID

    def save_experiment(
        self,
        expid: str,  # required
        # anything not none gets changed,
        # if delete_undistortion is true, undistortion is deleted
        delete_undistortion=False,
        timeframe_list: TimeFrameList = None,
        name: str = None,
        description: str = None,
        path: Path = None,
        regex: Regex = None,
        undistortion: Undistortion | None = None,
        plant_hsv_threshold: HSVThreshold = None,
        plant_rois: RoiList = None,
    ):
        exp: Experiment = self.load_experiment(expid)

        if delete_undistortion:
            exp.undistortion = None
        elif undistortion is not None:
            exp.undistortion = Undistortion

        if timeframe_list is not None:
            exp.timeframe_list = timeframe_list

        if name is not None:
            exp.name = name

        if description is not None:
            exp.description = description

        if path is not None:
            exp.path = path

        if regex is not None:
            exp.regex = regex

        if plant_hsv_threshold is not None:
            exp.plant_hsv_threshold = plant_hsv_threshold

        if plant_rois is not None:
            exp.plant_rois = plant_rois

        # delete the experiment
        self.delete_experiment(expid)

        # save the experiment
        exp.save(Path(self.experiment_dir_path) / expid)

    def delete_experiment(self, expid: str):
        shutil.rmtree(Path(self.experiment_dir_path) / expid)

    def run_experiment(self, expid: str):
        experiment = self.load_experiment(expid)
        p = Pipeline(experiment, self.experiment_dir_path)
        p.run()
