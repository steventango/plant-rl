from abc import abstractmethod
from typing import Any, Dict

from RlGlue.agent import BaseAgent as RlGlueBaseAgent


class BaseAgent(RlGlueBaseAgent):
    @abstractmethod
    def start(self, observation: Any) -> tuple[int, dict]:
        raise NotImplementedError('Expected `start` to be implemented')

    @abstractmethod
    def step(self, reward: float, observation: Any, extra: Dict[str, Any]) -> tuple[int, dict]:
        raise NotImplementedError('Expected `step` to be implemented')


class BasePlanningAgent(BaseAgent):
    @abstractmethod
    def plan(self) -> None:
        raise NotImplementedError("Expected `plan` to be implemented")
