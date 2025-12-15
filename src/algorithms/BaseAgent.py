from typing import Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

import utils.RlGlue.agent


class BaseAgent(utils.RlGlue.agent.BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector | None,
        seed: int,
    ):
        self.observations = observations
        self.actions = actions
        self.params = params
        self.collector = collector

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.gamma = params.get("gamma", 1)
        self.n_step = params.get("n_step", 1)
        self.use_planning = params.get("use_planning", False)

        # Observation indices to select specific dimensions
        self.observation_indices = params.get("observation_indices", None)

        # Ensure observations is a flat dimension
        if isinstance(observations, tuple):
            if len(observations) == 1:
                self.state_dim = observations[0]
            else:
                self.state_dim = int(np.prod(observations))
        else:
            self.state_dim = observations

        # If observation indices are provided, update state_dim
        if self.observation_indices is not None:
            self.state_dim = len(self.observation_indices)

    def process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Flatten and select specific dimensions of observation."""
        obs = obs.ravel()
        if self.observation_indices is not None:
            obs = obs[self.observation_indices]
        return obs

    def cleanup(self): ...

    def plan(self):
        if not self.use_planning:
            return

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        return {
            "__args": (
                self.observations,
                self.actions,
                self.params,
                self.collector,
                self.seed,
            ),
            "rng": self.rng,
        }

    def __setstate__(self, state):
        self.__init__(*state["__args"])
        self.rng = state["rng"]
