import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.tc.TCAgent import TCAgent
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities


@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha, z, lambda_, l1, l2):
    qsa = np.dot(w[a],x)

    qsp = np.dot(w, xp)

    delta = r + gamma * np.dot(qsp,pi) - qsa

    z *= gamma * lambda_
    z[a] += x

    w[a] = w[a] + (alpha / np.count_nonzero(x)) * delta * z[a] - l1 * np.sign(w[a]) - l2 * w[a]

@njit(cache=True)
def value(w, x):
    return np.dot(w,x)

@checkpointable(('w', 'z'))
class ESARSA(TCAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.lambda_ = params.get('lambda', 0.0)
        self.w0 = params.get('w0', 0.0)
        self.l1 = params.get('l1', 0.0)
        self.l2 = params.get('l2', 0.0)

        # create initial weights
        self.w = np.full((actions, self.n_features), self.w0, dtype=np.float64)
        self.z = np.zeros((actions, self.n_features), dtype=np.float64)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        return egreedy_probabilities(qs, self.actions, self.epsilon)

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)
            pi = np.zeros(self.actions)
            self.z = np.zeros_like(self.z)
        else:
            pi = self.policy(xp)

        _update(self.w, x, a, xp, pi, r, gamma, self.alpha, self.z, self.lambda_, self.l1, self.l2)
