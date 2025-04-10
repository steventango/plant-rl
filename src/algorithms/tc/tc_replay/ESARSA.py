import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.tc.tc_replay.TCAgentReplay import TCAgentReplay
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities

@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha):
    qsa = (x * w[a]).sum(axis=1)
    qsp = np.matmul(xp, w.T)
    delta = r + gamma * (qsp * pi).sum(axis=1) - qsa

    # TODO rescale learning rate by num active features
    grad = x * delta[:, None]
    np.add.at(w, a, alpha*grad)

@njit(cache=True)
def value(w, x):
    qs = np.matmul(x, w.T)
    return qs

@checkpointable(('w', ))
class ESARSA(TCAgentReplay):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.w = np.zeros((actions, self.n_features), dtype=np.float64)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        return egreedy_probabilities(qs, self.actions, self.epsilon)

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return
        
        # wait till batch size samples have been collected
        if self.buffer.size() <= self.batch_size:
            return
        
        self.updates += 1

        batch = self.buffer.sample(self.batch_size)
        pi = self.policy(batch.xp)
        _update(self.w, batch.x, batch.a, batch.xp, pi, batch.r, batch.gamma, self.alpha)
        
        self.buffer.update_batch(batch)

