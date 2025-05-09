import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.tc.TCAgent import TCAgent
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities

import logging
logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)

@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha, z, lambda_):
    qsa = np.dot(w[a],x)

    qsp = np.dot(w, xp)

    delta = r + gamma * np.dot(qsp,pi) - qsa

    z *= gamma * lambda_
    z[a] += x
    z[a] = np.minimum(z[a], 1.0)

    w += (alpha / np.count_nonzero(x)) * delta * z
    return delta


@njit(cache=True)
def value(w, x):
    return np.dot(w,x)

@checkpointable(('w', 'z'))
class ESARSA(TCAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.epsilon = params['epsilon']
        self.lambda_ = params.get('lambda', 0.0)
        self.w0 = params.get('w0', 0.0) / self.nonzero_features

        # create initial weights
        self.w = np.full((actions, self.n_features), self.w0, dtype=np.float64)
        self.z = np.zeros((actions, self.n_features), dtype=np.float64)

        self.info = {
            'w': self.w,
            'z': self.z,
        }
        if len(observations) == 1:
            self.all_obs = np.eye(self.tile_coder.features())
        else:
            self.all_obs = None


    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        self.info['qs'] = qs
        pi = egreedy_probabilities(qs, self.actions, self.epsilon)
        self.info['pi'] = pi
        if self.all_obs is not None:
            q = self.w @ self.all_obs
            self.info['q'] = q
            # calculate advantage
            advantage = q - pi @ q
            self.info['advantage'] = advantage
        return pi

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)
            pi = np.zeros(self.actions)
        else:
            pi = self.policy(xp)

        delta = _update(self.w, x, a, xp, pi, r, gamma, self.alpha, self.z, self.lambda_)
        if xp is None:
            self.z = np.zeros_like(self.z)

        self.info.update({
            'delta': delta,
            'w': self.w,
            'z': self.z,
            'x': x,
            'xp': xp,
        })

    def get_info(self):
        return self.info
