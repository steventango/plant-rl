from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import (
    OfflinePlantGrowthChamber as Env,
)
from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import (
    OfflinePlantGrowthChamber_1hrStep,
    OfflinePlantGrowthChamber_1hrStep_MC,
    OfflinePlantGrowthChamber_1hrStep_MC_Area_Openness,
    OfflinePlantGrowthChamber_1hrStep_MC_OpennessOnly,
    OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly,
    OfflinePlantGrowthChamberTime,
    OfflinePlantGrowthChamberTimeArea,
    OfflinePlantGrowthChamberTimeDLI,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class OfflinePlantGrowthChamber(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.actions = 2

        if self.env_params["type"] == "default":
            self.env = Env(**self.env_params)
            self.observations = (4,)
        elif self.env_params["type"] == "1hrStep":
            self.env = OfflinePlantGrowthChamber_1hrStep(**self.env_params)
            self.observations = (4,)
        elif self.env_params["type"] == "1hrStep_MC":
            self.env = OfflinePlantGrowthChamber_1hrStep_MC(**self.env_params)
            self.observations = (4,)
        elif self.env_params["type"] == "1hrStep_MC_OpennessOnly":
            self.env = OfflinePlantGrowthChamber_1hrStep_MC_OpennessOnly(
                **self.env_params
            )
            self.observations = (2,)
        elif self.env_params["type"] == "1hrStep_MC_TimeOnly":
            self.env = OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly(**self.env_params)
            self.observations = (2,)
        elif self.env_params["type"] == "1hrStep_MC_Area_Openness":
            self.env = OfflinePlantGrowthChamber_1hrStep_MC_Area_Openness(
                **self.env_params
            )
            self.observations = (2,)
        elif self.env_params["type"] == "Time":
            self.env = OfflinePlantGrowthChamberTime(**self.env_params)
            self.observations = (1,)
        elif self.env_params["type"] == "TimeArea":
            self.env = OfflinePlantGrowthChamberTimeArea(**self.env_params)
        elif self.env_params["type"] == "TimeDLI":
            self.env = OfflinePlantGrowthChamberTimeDLI(**self.env_params)
            self.observations = (2,)
        else:
            raise ValueError("Env type invalid.")
