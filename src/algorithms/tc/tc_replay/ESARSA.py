import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.tc.tc_replay.TCAgentReplay import TCAgentReplay
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities

import logging
logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)

@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha, num_active_features):
    logger.info(a)
    qsa = (x * w[a]).sum(axis=1)
    qsp = np.matmul(xp, w.T)
    delta = r + gamma * (qsp * pi).sum(axis=1) - qsa

    grad = x * delta[:, None]
    np.add.at(w, a, alpha / num_active_features * grad)  # TODO: Check if it makes sense to divide by num_active_features
    
    return delta

@njit(cache=True)
def value(w, x):
    qs = np.matmul(x, w.T)
    return qs

@checkpointable(('w', ))
class ESARSA(TCAgentReplay):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.epsilon = params['epsilon']
        self.w0 = params.get('w0', 0.0) / self.nonzero_features

        # create initial weights
        self.w = np.zeros((actions, self.n_features), dtype=np.float64)

        self.info = {
            'w': self.w
        }

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        self.info['qs'] = qs
        pi = egreedy_probabilities(qs, self.actions, self.epsilon)
        self.info['pi'] = pi
        return pi
    
    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def batch_update(self):
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
        delta = _update(self.w, batch.x, batch.a, batch.xp, pi, batch.r, batch.gamma, self.alpha, self.nonzero_features)
        
        self.info.update({
            'delta': delta,
            'w': self.w,
        })

        self.buffer.update_batch(batch)

        # (Optional) at the end of each episode, decay step size linearly
        if self.alpha_decay: 
            self.alpha = self.get_step_size()

    def get_info(self):
        return self.info
    
    def get_step_size(self):  # linear decay with minimum 
        min_alpha = 0.01
        horizon = 5e4
        return max(min_alpha, self.alpha0 * (1 - self.steps / horizon))

