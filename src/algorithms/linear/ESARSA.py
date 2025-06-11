import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.linear.LinearAgent import LinearAgent
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities


@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha, z, lambda_):
    qsa = np.dot(w[a],x)

    qsp = np.dot(w, xp)

    delta = r + gamma * np.dot(qsp,pi) - qsa

    z *= gamma * lambda_
    z[a] += x
    # z[a] = np.minimum(z[a], 1.0)

    w += (alpha / np.count_nonzero(x)) * delta * z

    return delta

@njit(cache=True)
def value(w, x):
    return np.dot(w,x)

@checkpointable(('w', 'z'))
class ESARSA(LinearAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.lambda_ = params.get('lambda', 0.0)

        # create initial weights, set to 0
        self.w = np.zeros((actions, self.observations[0]), dtype=np.float64)
        self.z = np.zeros((actions, self.observations[0]), dtype=np.float64)

        self.info = {
            'w': self.w,
            'z': self.z,
        }

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        self.info['qs'] = qs
        pi = egreedy_probabilities(qs, self.actions, self.epsilon)
        advantage = qs - pi @ qs
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
            'pi': pi,
        })
