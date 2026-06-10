from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, DIM_ACTION, RED_ACTION


class ActionSpec(ABC):
    name: str
    n_actions: int
    trace_dim: int = 6

    @abstractmethod
    def decode(self, action: Any, backend: Any) -> np.ndarray:
        pass

    def trace_action(self, action: Any, backend: Any) -> np.ndarray:
        """Convert an action to trace_dim for UEMA updates."""
        return self.decode(action, backend)


@dataclass(frozen=True)
class PPFD6Action(ActionSpec):
    name: str = "ppfd6"
    n_actions: int = 6
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        return np.asarray(action, dtype=np.float64)


@dataclass(frozen=True)
class IntensityAction(ActionSpec):
    name: str = "intensity"
    n_actions: int = 1
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        if isinstance(action, np.ndarray) and action.ndim > 0:
            return np.asarray(action, dtype=np.float64)
        return BALANCED_ACTION_105 * float(action)


@dataclass(frozen=True)
class DiscreteAction(ActionSpec):
    name: str = "discrete"
    n_actions: int = 2
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        if isinstance(action, np.ndarray):
            return np.asarray(action, dtype=np.float64)
        action_map = {0: DIM_ACTION, 1: BALANCED_ACTION_105}
        return action_map[int(action)]


@dataclass(frozen=True)
class ColorAction(ActionSpec):
    name: str = "color"
    n_actions: int = 3
    trace_dim: int = 6

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        if isinstance(action, np.ndarray):
            return np.asarray(action, dtype=np.float64)
        action_map = {
            0: BALANCED_ACTION_105,
            1: BLUE_ACTION,
            2: RED_ACTION,
        }
        return action_map[int(action)]


@dataclass(frozen=True)
class ColorTriangleAction(ActionSpec):
    name: str = "color_triangle"
    n_actions: int = 3
    trace_dim: int = 3

    def decode(self, action: Any, backend: Any) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] == 6:
            return action
        basis = np.column_stack([RED_ACTION, BALANCED_ACTION_105, BLUE_ACTION])
        return basis @ action

    def trace_action(self, action: Any, backend: Any) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] == self.trace_dim:
            return action
        if action.shape[0] == 6:
            basis = np.column_stack([RED_ACTION, BALANCED_ACTION_105, BLUE_ACTION])
            return np.linalg.lstsq(basis, action, rcond=None)[0]
        raise ValueError(
            f"Expected action with {self.trace_dim} or 6 elements, got {action.shape[0]}"
        )


ACTION_SPECS: dict[str, ActionSpec] = {
    "ppfd6": PPFD6Action(),
    "intensity": IntensityAction(),
    "discrete": DiscreteAction(),
    "color": ColorAction(),
    "color_triangle": ColorTriangleAction(),
}
