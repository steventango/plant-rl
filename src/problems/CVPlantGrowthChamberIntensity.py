from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.CVPlantGrowthChamberIntensity import CVPlantGrowthChamberIntensity as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class CVPlantGrowthChamberBinary(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 6
        self.observations = (6,)
        self.gamma = 0.99
