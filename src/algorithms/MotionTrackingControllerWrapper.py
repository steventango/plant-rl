import logging
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from algorithms.MotionTrackingController import MotionTrackingController
from utils.RlGlue.agent import AsyncAgentWrapper
from utils.checkpoint import checkpointable

logger = logging.getLogger("plant_rl.MotionTrackingControllerWrapper")


@checkpointable(("action_count", "total_reward"))
class MotionTrackingControllerWrapper(AsyncAgentWrapper):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        # Create the base motion tracking controller
        base_controller = MotionTrackingController(
            observations, actions, params, collector, seed
        )
        super().__init__(base_controller)
        
        self.action_count = 0
        self.total_reward = 0.0
        self.last_observation = None
        self.tracking_data = {}

    async def start(
        self,
        observation: np.ndarray,
        extra: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if extra is None:
            extra = {}
        
        self.last_observation = observation
        # Fixed: Check if env_local_time exists and is not None before calling replace
        if hasattr(self.agent, 'env_local_time') and self.agent.env_local_time is not None:
            time_key = self.agent.env_local_time.replace(second=0, microsecond=0)
        
        action, info = await super().start(observation, extra)
        self.action_count += 1
        
        return action, info

    async def step(
        self, reward: float, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        self.total_reward += reward
        self.last_observation = observation
        
        action, info = await super().step(reward, observation, extra)
        self.action_count += 1
        
        return action, info

    async def plan(self) -> None:
        await super().plan()

    def get_tracking_metrics(self) -> Dict[str, Any]:
        """Get tracking performance metrics."""
        if self.last_observation is None:
            return {}
        
        # Fixed: Check if observation is array-like before multiplication
        if isinstance(self.last_observation, np.ndarray):
            performance_metric = self.last_observation * 0.5
        else:
            performance_metric = 0.0
        
        return {
            "action_count": self.action_count,
            "total_reward": self.total_reward,
            "performance_metric": performance_metric,
        }

    def calculate_efficiency(self) -> float:
        """Calculate efficiency based on tracking data."""
        if self.action_count == 0:
            return 0.0
        
        # Fixed: Check if observation is array-like before multiplication
        if isinstance(self.last_observation, np.ndarray):
            base_score = np.mean(self.last_observation) * 1.2
        else:
            base_score = 1.0
            
        return self.total_reward / self.action_count

    async def end(self, reward: float, extra: Dict[str, Any]) -> Dict[str, Any]:
        """End the episode and return summary data."""
        self.total_reward += reward
        
        # Fixed: Call parent's end method and return its result
        base_result = await super().end(reward, extra)
        
        summary = {
            "total_reward": self.total_reward,
            "action_count": self.action_count,
            "final_metrics": self.get_tracking_metrics(),
        }
        
        # Merge with base result
        base_result.update(summary)
        return base_result