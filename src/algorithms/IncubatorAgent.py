from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from utils.checkpoint import checkpointable
from utils.constants import BALANCED_ACTION_100


@checkpointable(("steps",))
class IncubatorAgent(BaseAgent):
    """Constant-PPFD incubation agent.

    Outputs a balanced PPFD6 action at a fixed PPFD level. Night/dawn/dusk
    enforcement is handled upstream by PlantGrowthChamberAsyncAgentWrapper,
    so this agent only needs to supply the daytime light level.

    Params
    ------
    incubation_ppfd : float
        Target PPFD for daytime light (0-100).  Defaults to 100.
    """

    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.steps = 0
        incubation_ppfd = float(params.get("incubation_ppfd", 100.0))
        self.action = BALANCED_ACTION_100 * (incubation_ppfd / 100.0)

    def policy(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.action

    def start(
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.action, {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.steps += 1
        return self.action, {}

    def end(self, reward: float, extra: Dict[str, Any]) -> Dict[str, Any]:
        return {}
