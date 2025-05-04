from abc import abstractmethod
from typing import Any, Dict, Tuple


class BaseAsyncEnvironment:
    @abstractmethod
    async def start(self) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("Expected `start` to be implemented")

    @abstractmethod
    async def step(self, action: Any) -> Tuple[float, Any, bool, Dict[str, Any]]:
        raise NotImplementedError("Expected `step` to be implemented")
