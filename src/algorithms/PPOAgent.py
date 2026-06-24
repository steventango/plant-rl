from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

# Must be set before torch/triton are imported to prevent segfault on machines
# with incompatible CUDA drivers (triton probes the driver at import time).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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
        print(f"[PPOAgent] Loading model from {frozen_agent_path!r} on cpu", flush=True)
        self.model = PPO.load(frozen_agent_path, device="cpu")
        print(f"[PPOAgent] Model loaded successfully (policy: {self.model.policy})", flush=True)

    def policy(self, obs: np.ndarray) -> Tuple[Any, Dict]:
        obs = self.process_observation(obs)
        logger.debug(f"Predicting action for obs {obs}")
        action, _ = self.model.predict(obs, deterministic=True)
        logger.debug(f"Action: {action}")
        return action, {}

    def start(self, observation: np.ndarray, extra: Dict | None = None) -> Tuple[Any, Dict]:
        print(f"[PPOAgent] start obs={observation}", flush=True)
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
        print(f"[PPOAgent] Restoring model from {state['frozen_agent_path']!r}", flush=True)
        self.model = PPO.load(state["frozen_agent_path"], device="cpu")
