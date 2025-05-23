from ml_instrumentation.Collector import Collector

from environments.PlantGrowthChamber.CVPlantGrowthChamber import CVPlantGrowthChamber as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseAsyncProblem import BaseAsyncProblem


class CVPlantGrowthChamber(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 6
        self.observations = (1,)
