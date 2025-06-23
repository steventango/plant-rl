import asyncio
from datetime import datetime, timedelta
from typing import Any

from utils.RlGlue.agent import AsyncAgentWrapper, BaseAgent


class PlantGrowthChamberAsyncAgentWrapper(AsyncAgentWrapper):
    def __init__(self, agent: BaseAgent):
        super().__init__(agent)

    async def start(self, observation: Any, extra: dict[str, Any] = {}) -> tuple[Any, dict[str, Any]]:
        self.last_action_info = await asyncio.to_thread(self.agent.start, observation, extra)
        return self.last_action_info

    async def step(self, reward: float, observation: Any, extra: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self.get_next_step_time(self.duration) - datetime.now() < self.duration:
            return self.last_action_info
        self.last_action_info = await asyncio.to_thread(self.agent.step, reward, observation, extra)
        return self.last_action_info

    def get_next_step_time(self, duration: timedelta):
        duration_s = duration.total_seconds()
        wake_time = datetime.fromtimestamp((datetime.now().timestamp() // duration_s + 1) * duration_s)
        return wake_time

    async def end(self, reward: float, extra: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.agent.end, reward, extra)
