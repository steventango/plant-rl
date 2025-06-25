from typing import Dict, Tuple

import numpy as np
from numba import njit
from PyExpUtils.collection.Collector import Collector

from algorithms.tc.TCAgent import TCAgent
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities


@njit(cache=True)
def _update(w, x, a, xp, r, gamma, alpha):
    qsa = np.dot(w[a], x)

    qsp = np.dot(w, xp).max()

    delta = r + gamma * qsp - qsa

    w[a] = w[a] + alpha * delta * x


@njit(cache=True)
def value(w, x):
    return np.dot(w, x)


@checkpointable(("w",))
class QL(TCAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]

        # create initial weights
        self.w = np.zeros((actions, self.n_features), dtype=np.float64)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        return egreedy_probabilities(qs, self.actions, self.epsilon)

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)

        _update(self.w, x, a, xp, r, gamma, self.alpha)
