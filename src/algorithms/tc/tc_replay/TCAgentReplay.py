from abc import abstractmethod  # type: ignore
from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.utils.random import sample
from ReplayTables.interface import Timestep
from ReplayTables.registry import build_buffer

from algorithms.tc.TCAgent import TCAgent
from utils.checkpoint import checkpointable


@checkpointable(("buffer", "steps", "updates"))
class TCAgentReplay(TCAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        # Params for replay
        self.buffer_size = params["buffer_size"]
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
    def start(  # type: ignore
        self, s: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.buffer.flush()

        x = self.get_rep(s)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.buffer.add_step(
            Timestep(
                x=x,
                a=a,
                r=None,
                gamma=self.gamma,
                terminal=False,
            )
        )
        return a, self.get_info()  # type: ignore

    def step(self, r: float, sp: np.ndarray | None, extra: Dict[str, Any]):
        a = -1

        # sample next action
        xp = None
        if sp is not None:
            xp = self.get_rep(sp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

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

        self.batch_update()
        return a, self.get_info()

    def end(self, r: float, extra: Dict[str, Any]):
        self.buffer.add_step(
            Timestep(
                x=np.zeros(self.n_features),
                a=-1,
                r=r,
                gamma=0,
                terminal=True,
            )
        )

        self.batch_update()

        return {}
