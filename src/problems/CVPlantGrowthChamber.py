from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class CVPlantGrowthChamber(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 6
        self.observations = (2,)
        self.gamma = 0.99
