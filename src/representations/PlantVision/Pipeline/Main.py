from ExperimentManager import ExperimentManager
import argparse
from pathlib import Path


def create():
    mngr = ExperimentManager()
    experiment_id = mngr.create_experiment(
        name="DEMO TITLE",
        timeframe_list=[
            {"date": "2023-03-25", "ontime": "11:59", "offtime": "13:01"},
        ],
        description="DEMO DESCRIPTION",
        path=str(Path.cwd() / "TestImageDir"),
        regex="Date1",
        undistortion={
            "fx": "1800",
            "fy": "1800",
            "cx": "1296",
            "cy": "972",
            "k1": "0",
            "k2": "0",
            "k3": "0",
            "k4": "0",
        },
        threshold={
            "hl": "20",
            "hh": "80",
            "sl": "10",
            "sh": "255",
            "vl": "125",
            "vh": "255",
            "fill": "50",
            "invert": False,
        },
        rois=[{"cx":378.2551181102362,"cy":461.1728481455563,"r":68.03149606299212,"number":0,"genotype":"genotype1"},{"cx":587.7921259842519,"cy":444.8481455563331,"r":68.03149606299212,"number":1,"genotype":"genotype2"}],
    )
    print(f"Created Experiment {experiment_id}")


def list_experiments():
    mngr = ExperimentManager(True)


def run_experiment():
    mngr = ExperimentManager()
    s = input("Enter the experiment ID: ")
    print(f"Received '{s}'")
    mngr.run_experiment(s)


if __name__ == "__main__":
    current_dir = Path.cwd()
    subdir = "Experiments"
    full_path = current_dir / subdir
    print(full_path)
    parser = argparse.ArgumentParser(description="Experiment CLI tool")
    parser.add_argument("mode", choices=["create", "list", "run"])
    args = parser.parse_args()

    if args.mode == "create":
        print("Creating a new experiment...")
        create()

    elif args.mode == "list":
        print("Listing all experiments...")
        list_experiments()

    elif args.mode == "run":
        print("Running an experiment...")
        run_experiment()

    else:
        print("ERROR")
