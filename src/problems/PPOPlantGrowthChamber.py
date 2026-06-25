from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.factory import create_plant_growth_chamber
from environments.PlantGrowthChamber.specs import ACTION_SPECS, create_observation_spec
from experiment.ExperimentModel import ExperimentModel
from problems.BasePlantGrowthChamberAsyncProblem import (
    BasePlantGrowthChamberAsyncProblem,
)


class PPOPlantGrowthChamber(BasePlantGrowthChamberAsyncProblem):
    """Problem that runs a pre-trained SB3 PPO agent against the PlantGrowthChamber.

    Expected env params:
        action      : "continuous_color"
        observation : "day_area_color_trace"
        zone        : e.g. "alliance-zone01"
        timezone    : e.g. "Etc/GMT-2"
        policy_path  : path to the learned policy in .pt format
    """

    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        print("[PPOPlantGrowthChamber] __init__ start", flush=True)
        super().__init__(exp, idx, collector)
        print(f"[PPOPlantGrowthChamber] env_params={self.env_params}", flush=True)

        action_name = self.env_params.get("action", "continuous_color")
        observation_name = self.env_params.get("observation", "day_area_color_trace")
        print(f"[PPOPlantGrowthChamber] action={action_name!r} observation={observation_name!r}", flush=True)
        action_spec = ACTION_SPECS[action_name]
        observation_spec = create_observation_spec(
            observation_name, action_spec, self.env_params
        )
        print("[PPOPlantGrowthChamber] observation_spec created, creating environment...", flush=True)
        self.env = create_plant_growth_chamber(**self.env_params)
        print("[PPOPlantGrowthChamber] environment created", flush=True)
        self.observations = observation_spec.shape
        self.actions = action_spec.n_actions
        print(f"[PPOPlantGrowthChamber] observations={self.observations} actions={self.actions}", flush=True)
