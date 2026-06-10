from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.factory import create_plant_growth_chamber
from environments.PlantGrowthChamber.specs import ACTION_SPECS, create_observation_spec
from experiment.ExperimentModel import ExperimentModel
from problems.BasePlantGrowthChamberAsyncProblem import (
    BasePlantGrowthChamberAsyncProblem,
)


class PlantGrowthChamber(BasePlantGrowthChamberAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        observation_name = self.env_params.get("observation", "scalar")
        action_name = self.env_params.get("action", "ppfd6")
        action_spec = ACTION_SPECS[action_name]
        observation_spec = create_observation_spec(
            observation_name, action_spec, self.env_params
        )

        self.env = create_plant_growth_chamber(**self.env_params)
        self.observations = observation_spec.shape
        self.actions = action_spec.n_actions
