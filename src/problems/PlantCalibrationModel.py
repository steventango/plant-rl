from typing import Any, Dict, Tuple

import numpy as np

from plant_models import PlantCalibrationModel as Env
from PyExpUtils.collection.Collector import Collector
from RlGlue.environment import BaseEnvironment

from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class RlGlueWrapper(BaseEnvironment):
    def __init__(self, env):
        self.env = env

    def start(self) -> Tuple[Any, Dict[str, Any]]:
        s, info = self.env.reset()
        return np.asarray(s), info

    def step(self, action: Any) -> Tuple[float, Any, bool, Dict[str, Any]]:
        sp, r, t, trunc, info = self.env.step(action)
        return r, np.asarray(sp), t or trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


class PlantCalibrationModel(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env_params = self.params.get("environment", {})
        self.eval_env_params = self.params.get("eval_environment", {})

        # Instantiate the environment
        self.env = self._build_env(self.env_params)
        self.eval_env = self._build_env(self.eval_env_params)

        self.actions = self.env.env.action_space.shape[0]
        self.observations = self.env.env.observation_space.shape

        self.gamma = self.params.get("agent", {}).get("gamma", 0.99)

    def _build_env(self, params):
        gym_env = Env(**params)
        return RlGlueWrapper(gym_env)

    def getEvalEnvironment(self):
        return self.eval_env
