import asyncio  # type: ignore
from abc import abstractmethod
from typing import Any

from RlGlue.agent import BaseAgent as RlGlueBaseAgent


class BaseAgent(RlGlueBaseAgent):
    @abstractmethod
    def start(  # type: ignore
        self, observation: Any, extra: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("Expected `start` to be implemented")

    @abstractmethod
    def step(  # type: ignore
        self, reward: float, observation: Any, extra: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("Expected `step` to be implemented")

    @abstractmethod
    def plan(self) -> None:
        raise NotImplementedError("Expected `plan` to be implemented")

    @abstractmethod
    def end(self, reward: float, extra: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        raise NotImplementedError("Expected `end` to be implemented")


class BaseAsyncAgent:
    @abstractmethod
    async def start(
        self,
        observation: Any,
        extra: dict[str, Any] = None,  # type: ignore
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("Expected `start` to be implemented")

    @abstractmethod
    async def step(
        self, reward: float, observation: Any, extra: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("Expected `step` to be implemented")

    @abstractmethod
    async def plan(self) -> None:
        raise NotImplementedError("Expected `plan` to be implemented")

    @abstractmethod
    async def end(self, reward: float, extra: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Expected `end` to be implemented")


class AsyncAgentWrapper(BaseAsyncAgent):
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def start(
        self,
        observation: Any,
        extra: dict[str, Any] = None,  # type: ignore
    ) -> tuple[Any, dict[str, Any]]:
        if extra is None:
            extra = {}
        return await asyncio.to_thread(self.agent.start, observation, extra)

    async def step(
        self, reward: float, observation: Any, extra: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return await asyncio.to_thread(self.agent.step, reward, observation, extra)

    async def plan(self) -> None:
        return await asyncio.to_thread(self.agent.plan)

    async def end(self, reward: float, extra: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.agent.end, reward, extra)

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        return {"__args": (self.agent,)}

    def __setstate__(self, state):
        self.__init__(*state["__args"])
