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
    qsa = np.dot(w[a],x)
    assert qsa.shape == ()

    qsp = np.dot(w, xp)
    assert qsp.shape == (5,)
    
    assert np.dot(qsp,pi).shape == ()

    delta = r + gamma * np.dot(qsp,pi) - qsa

    w[a] = w[a] + alpha * delta * x
    
    return alpha * delta * x

#@njit(cache=True)
def value(w, x):
    return np.dot(w,x)

@checkpointable(('w', ))
class ESARSA(LinearAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights, set to -10
        self.w = np.zeros((actions, self.observations[0]), dtype=np.float64)*-10
        assert self.w.shape == (5, 9)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        return egreedy_probabilities(qs, self.actions, self.epsilon)

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        assert x.shape == (9,)
        return value(self.w, x)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)
            pi = np.zeros(self.actions)
        else:
            pi = self.policy(xp)
        
        update_term = _update(self.w, x, a, xp, pi, r, gamma, self.alpha)
        #self.info = {'x': np.argmax(x), 'a':a, 'delta': delta, 'pi': pi, 'action vals for state': self.w[:, np.argmax(x)], 'next': xp}
        policy = []
        for i in range(self.observations[0]):
            state = np.zeros(self.observations[0], dtype=np.float64)
            state[i] = 1
            policy.append(self.policy(state).round(3))
        policy = np.array(policy, dtype=np.float64)
        policy_str = "\n".join(" ".join(map(str, row)) for row in policy)      
        self.policy_str = policy_str
        self.info={'values':self.values(x), 'update_term':update_term, 'a':a, 'x':x, 'xp':xp, 'gamma':gamma}

