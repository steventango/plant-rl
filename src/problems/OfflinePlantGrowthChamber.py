from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.OfflinePlantGrowthChamber import OfflinePlantGrowthChamber as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class OfflinePlantGrowthChamber(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 2
        self.observations = (2,)
