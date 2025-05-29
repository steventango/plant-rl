from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import OfflinePlantGrowthChamber as Env
from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import OfflinePlantGrowthChamber_1hrStep
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class OfflinePlantGrowthChamber(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.observations = (2,)

        if self.env_params['type'] == 'default':
            self.env = Env(**self.env_params)
            self.actions = 2
        elif self.env_params['type'] == '1hrStep':
            self.env = OfflinePlantGrowthChamber_1hrStep(**self.env_params)
            self.actions = 2
        else:
            raise ValueError(f"Env type invalid.")
