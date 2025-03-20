import numpy as np

from abc import abstractmethod
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.utils.random import sample
from ReplayTables.interface import Timestep
from ReplayTables.ingress.LagBuffer import LagBuffer

from algorithms.BaseAgent import BaseAgent
from representations.TileCoder import SparseTileCoder, TileCoderConfig
from utils.checkpoint import checkpointable

@checkpointable(('rep', 'lag'))
class TCAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.lag = LagBuffer(self.n_step)

        self.rep_params: Dict = params['representation']
        assert self.rep_params['tc_dim'] <= observations[0], "Error: tc_dim > state dimension!"

        # NOTE: Only the LAST tc_dim features are tile-coded. Would like to come back and make this more elegent but we are trying to move fast
        self.tc_dim = self.observations[0] if self.rep_params['tc_dim'] is None else self.rep_params['tc_dim']

        self.tile_coder = SparseTileCoder(TileCoderConfig(
            tiles=self.rep_params['tiles'],
            tilings=self.rep_params['tilings'],
            dims=self.tc_dim,
            input_ranges=None,    # assume that inputs are in the range (0.0, 1.0)
        ))
    
        self.n_features = self.observations[0] - self.tc_dim + self.tile_coder.features()

    def get_rep(self, s):
        # only the last "tc_dim" inputs are tile coded
        if self.tc_dim > 0: 
            x = np.concatenate((s[:-self.tc_dim], self.tile_coder.encode(s[-self.tc_dim:])))
        else: 
            x = s
        return x

    @abstractmethod
    def policy(self, obs: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, x, a, xp, r, gamma):
        ...

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, s: np.ndarray):
        self.lag.flush()
        x = self.get_rep(s)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.lag.add(Timestep(
            x=x,
            a=a,
            r=None,
            gamma=0,
            terminal=False,
        ))
        return a

    def step(self, r: float, sp: np.ndarray | None, extra: Dict[str, Any]):
        a = -1

        # sample next action
        xp = None
        if sp is not None:
            xp = self.get_rep(sp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get('gamma', 1.0)

        interaction = Timestep(
            x=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
            terminal=False,
        )

        for exp in self.lag.add(interaction):
            self.update(
                x=exp.x,
                a=exp.a,
                xp=exp.n_x,
                r=exp.r,
                gamma=exp.gamma,
            )

        return a

    def end(self, r: float, extra: Dict[str, Any]):
        interaction = Timestep(
            x=None,
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
        )
        for exp in self.lag.add(interaction):
            self.update(
                x=exp.x,
                a=exp.a,
                xp=exp.n_x,
                r=exp.r,
                gamma=exp.gamma,
            )

        self.lag.flush()
