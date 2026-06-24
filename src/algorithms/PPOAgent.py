from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector
from stable_baselines3 import PPO

from algorithms.BaseAgent import BaseAgent


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
        self.model = PPO.load(frozen_agent_path)

    def policy(self, obs: np.ndarray) -> Tuple[Any, Dict]:
        obs = self.process_observation(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return action, {}

    def start(self, observation: np.ndarray, extra: Dict | None = None) -> Tuple[Any, Dict]:
        return self.policy(observation)

    def step(self, reward: float, observation: np.ndarray, extra: Dict | None = None) -> Tuple[Any, Dict]:
        return self.policy(observation)

    # Checkpointing: model weights live in the .zip file, not in agent state.
    def __getstate__(self):
        state = super().__getstate__()
        state["frozen_agent_path"] = self.params["frozen_agent_path"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = PPO.load(state["frozen_agent_path"])
