import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.linear.LinearAgent import LinearAgent
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


#@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha):
    qsa = w[a].dot(x)

    qsp = w.dot(xp)

    delta = r + gamma * qsp.dot(pi) - qsa

    w[a] = w[a] + alpha * delta

#@njit(cache=True)
def value(w, x):
    return w.dot(x)

@checkpointable(('w', ))
class ESARSA(LinearAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.w = np.zeros((actions, self.observations[0]), dtype=np.float64)

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
        else:
            pi = self.policy(xp)
        self.info = {'x': np.argmax(x), 'pi': pi, 'a':a, 'action vals for state': self.w[:, np.argmax(x)], 'term': xp==None}



        _update(self.w, x, a, xp, pi, r, gamma, self.alpha)
