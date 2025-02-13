from abc import abstractmethod
from typing import Any, Dict, Tuple


class BaseAsyncEnvironment:
    @abstractmethod
    def start(self) -> Any:
        raise NotImplementedError("Expected `start` to be implemented")

    @abstractmethod
    def step_one(self, action: Any) -> None:
        raise NotImplementedError("Expected `step_one` to be implemented")

    @abstractmethod
    def step_two(self) -> Tuple[float, Any, bool, Dict[str, Any]]:
        raise NotImplementedError("Expected `step_two` to be implemented")
