import numpy as np

from abc import abstractmethod
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.utils.random import sample
from ReplayTables.interface import Timestep
from ReplayTables.ingress.LagBuffer import LagBuffer

from algorithms.BaseAgent import BaseAgent
from representations.TileCoder import DenseTileCoder, TileCoderConfig
from representations.RichTileCoder import RichTileCoder, RichTileCoderConfig
from utils.checkpoint import checkpointable

@checkpointable(('lag',))
class TCAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.lag = LagBuffer(self.n_step)

        self.rep_params: Dict = params['representation']

        if self.rep_params['which_tc'] == 'RichTileCoder':
            self.tile_coder = RichTileCoder(RichTileCoderConfig(
                tiles=self.rep_params['tiles'],
                tilings=self.rep_params['tilings'],
                dims=observations[0],
                strategy=self.rep_params['strategy'],
            ))
        elif self.rep_params['which_tc'] == 'AndyTileCoder':
            self.tile_coder = DenseTileCoder(TileCoderConfig(
                tiles=self.rep_params['tiles'],
                tilings=self.rep_params['tilings'],
                dims=observations[0],
            ))
        else:
            raise ValueError(f"Please specify which tile coder to use with param which_tc.")

        self.n_features = self.tile_coder.features()
        self.nonzero_features = self.tile_coder.nonzero_features()

        self.alpha = params['alpha']
        self.alpha0 = params['alpha']
        self.alpha_decay = params.get('alpha_decay', False)

    def get_rep(self, s):
        return self.tile_coder.encode(s)

    def get_info(self):
        return {}

    @abstractmethod
    def policy(self, obs: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, x, a, xp, r, gamma):
        ...

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, s: np.ndarray, extra: Dict[str, Any] = {}):
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
        return a, self.get_info()

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

        return a, self.get_info()

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

        return {}
