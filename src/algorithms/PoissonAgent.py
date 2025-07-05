import logging
from typing import Any, Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent

logger = logging.getLogger("PoissonAgent")
logger.setLevel(logging.DEBUG)


class PoissonAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        self.updates = 0
        self.lam = params.get("lam", 3.0)
        self.max_repeat = params.get("max_repeat", 5)
        assert self.lam > 0, "Parameter 'lam' (lambda) must be positive."
        self.actions = actions
        self.current_action = None
        self.current_repeat = None

    def sample_action(self):
        # Sample an action uniformly
        action = self.rng.integers(0, self.actions)
        # Sample the number of repetitions from Poisson
        repeat = min(self.rng.poisson(self.lam), self.max_repeat)
        # Store in state
        self.current_action = action
        self.current_repeat = repeat
        logger.info(f"Sampled action: {action}, repeat: {repeat}")
        return action

    # ----------------------
    # -- RLGlue interface --
    # ----------------------

    def start(  # type: ignore
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        self.sample_action()
        assert self.current_action is not None
        logger.info(
            f"Start: action={self.current_action}, repeat={self.current_repeat}"
        )
        return self.current_action, {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        # Decrement repeat count, only sample new action if repeat is 0
        assert self.current_repeat is not None
        self.current_repeat -= 1
        if self.current_repeat < 1:
            # Sample a new action
            self.sample_action()
        logger.info(f"Step: action={self.current_action}, repeat={self.current_repeat}")
        return self.current_action, {}

    def end(self, reward: float, extra: Dict[str, Any]):
        # Optionally clear state
        self.current_action = None
        self.current_repeat = None
        return {}
