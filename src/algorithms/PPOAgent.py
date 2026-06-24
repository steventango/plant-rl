from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector
from stable_baselines3 import PPO

from algorithms.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """Wraps a pre-trained Stable-Baselines3 PPO model as a BaseAgent.

    Expects params["frozen_agent_path"] pointing to the .zip file produced by PPO.save().
    The model is loaded once at construction; start/step just call predict().
    """

    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector | None,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        frozen_agent_path: str = params["frozen_agent_path"]
        logger.info(f"Loading PPO model from {frozen_agent_path!r} on cpu")
        self.model = PPO.load(frozen_agent_path, device="cpu")
        logger.info(f"PPO model loaded successfully (policy: {self.model.policy})")

    def policy(self, obs: np.ndarray) -> Tuple[Any, Dict]:
        obs = self.process_observation(obs)
        logger.debug(f"Predicting action for obs {obs}")
        action, _ = self.model.predict(obs, deterministic=True)
        logger.debug(f"Action: {action}")
        return action, {}

    def start(self, observation: np.ndarray, extra: Dict | None = None) -> Tuple[Any, Dict]:
        logger.info(f"Agent start, obs shape={np.shape(observation)}, obs={observation}")
        return self.policy(observation)

    def step(self, reward: float, observation: np.ndarray, extra: Dict | None = None) -> Tuple[Any, Dict]:
        logger.debug(f"Agent step, reward={reward}, obs={observation}")
        return self.policy(observation)

    # Checkpointing: model weights live in the .zip file, not in agent state.
    def __getstate__(self):
        state = super().__getstate__()
        state["frozen_agent_path"] = self.params["frozen_agent_path"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        logger.info(f"Restoring PPO model from {state['frozen_agent_path']!r}")
        self.model = PPO.load(state["frozen_agent_path"], device="cpu")
