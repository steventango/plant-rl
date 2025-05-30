from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import OfflinePlantGrowthChamber as Env
from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import OfflinePlantGrowthChamber_1hrStep, OfflinePlantGrowthChamber_1hrStep_MC, OfflinePlantGrowthChamber_1hrStep_MC_AreaOnly, OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class OfflinePlantGrowthChamber(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.observations = (2,)
        self.actions = 2

        if self.env_params['type'] == 'default':
            self.env = Env(**self.env_params)
        elif self.env_params['type'] == '1hrStep':
            self.env = OfflinePlantGrowthChamber_1hrStep(**self.env_params)
        elif self.env_params['type'] == '1hrStep_MC':
            self.env = OfflinePlantGrowthChamber_1hrStep_MC(**self.env_params)
        elif self.env_params['type'] == '1hrStep_MC_AreaOnly':
            self.env = OfflinePlantGrowthChamber_1hrStep_MC_AreaOnly(**self.env_params)
        elif self.env_params['type'] == '1hrStep_MC_TimeOnly':
            self.env = OfflinePlantGrowthChamber_1hrStep_MC_TimeOnly(**self.env_params)
        else:
            raise ValueError(f"Env type invalid.")
