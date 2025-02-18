from abc import abstractmethod

from RlGlue.agent import BaseAgent


class BasePlanningAgent(BaseAgent):
    @abstractmethod
    def plan(self) -> None:
        raise NotImplementedError("Expected `plan` to be implemented")
