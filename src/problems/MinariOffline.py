import minari
from PyExpUtils.collection.Collector import Collector

from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class MinariOffline(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        dataset_name = self.exp_params["dataset"]
        self.dataset = minari.load_dataset(dataset_name)

        self.observations = self.dataset.observation_space.shape
        self.actions = self.dataset.action_space.shape[0]

        self.gamma = self.params.get("agent", {}).get("gamma", 0.99)
