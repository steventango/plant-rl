from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

# Must be set before torch/triton are imported to prevent segfault on machines
# with incompatible CUDA drivers (triton probes the driver at import time).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """Wraps a pre-trained SB3 PPO policy as a BaseAgent.

    Expects params["policy_path"] pointing to the .pt file produced by
    torch.save(agent.policy, ...) in train_agent.py.
    The policy is loaded once at construction; start/step just call predict().
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
        policy_path: str = params["policy_path"]
        print(f"[PPOAgent] Loading policy from {policy_path!r} on cpu", flush=True)
        self.model = torch.load(policy_path, map_location="cpu", weights_only=False)
        self.model.set_training_mode(False)
        print(f"[PPOAgent] Policy loaded successfully ({self.model})", flush=True)

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