from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Timestep
from ReplayTables.registry import build_buffer

from algorithms.tc.TCAgent import TCAgent
from utils.checkpoint import checkpointable


@checkpointable(("buffer", "steps", "updates"))
class TCAgentOffline(TCAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # Params for replay
        self.buffer_size = params["buffer_size"]
        self.batch = params["batch"]
        self.batch_size = params["batch"]
        self.update_freq = params.get("update_freq", 1)
        self.updates_per_step = params.get("updates_per_step", 1)

        self.buffer = build_buffer(
            buffer_type=params["buffer_type"],
            max_size=self.buffer_size,
            lag=self.n_step,
            rng=self.rng,
            config=params.get("buffer_config", {}),
        )

        self.steps = 0
        self.updates = 0

    @abstractmethod
    def batch_update(self): ...

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def load_start(self, s: np.ndarray, extra: Dict[str, Any]):
        self.buffer.flush()

        x = self.get_rep(s)
        a = extra["action"]
        self.buffer.add_step(
            Timestep(
                x=x,
                a=a,
                r=None,
                gamma=self.gamma,
                terminal=False,
            )
        )
        return {}

    def load_step(self, r: float, sp: np.ndarray | None, extra: Dict[str, Any]):
        a = extra["action"]

        # sample next action
        xp = None
        if sp is not None:
            xp = self.get_rep(sp)

        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        self.buffer.add_step(
            Timestep(
                x=xp,
                a=a,
                r=r,
                gamma=self.gamma * gamma,
                terminal=False,
            )
        )

        return {}

    def load_end(self, r: float, extra: Dict[str, Any]):
        self.buffer.add_step(
            Timestep(
                x=np.zeros(self.n_features),
                a=-1,
                r=r,
                gamma=0,
                terminal=True,
            )
        )

        return {}

    def plan(self):
        self.batch_update()
        return self.get_info()