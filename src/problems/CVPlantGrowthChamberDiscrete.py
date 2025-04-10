from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.CVPlantGrowthChamberDiscrete import CVPlantGrowthChamberDiscrete as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class CVPlantGrowthChamberDiscrete(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 4
        self.observations = (2,)
        self.gamma = 0.99
