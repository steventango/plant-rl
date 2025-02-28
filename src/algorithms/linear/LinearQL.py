from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.utils.random import sample

from algorithms.BaseAgent import BaseAgent
from utils.policies import egreedy_probabilities


class LinearQL(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.curr_state = None
        self.curr_action = None

        # create initial weights, set to -10
        self.w = np.zeros((actions, self.observations[0]), dtype=np.float64)
    
    def get_egreedy_action(self, s):
        qs = np.dot(self.w, s)
        probs = egreedy_probabilities(qs, self.actions, self.epsilon)
        a = sample(probs, rng=self.rng)
        return a
    
    def update(self, r, s_next):
        next_q = np.dot(self.w, s_next).max()
        delta = r + self.gamma*next_q - np.dot(self.w[self.curr_action], self.curr_state)
        self.w[self.curr_action] += self.alpha * delta * self.curr_state

    def start(self, s: np.ndarray):
        a = self.get_egreedy_action(s)
        self.curr_state = s
        self.curr_action = a
        return a

    def step(self, r: float, s_next, extra):
        self.update(r, s_next)
        a = self.get_egreedy_action(s_next)
        self.curr_action = a
        self.curr_state = s_next
        # Decay epsilon
        #self.epsilon = max(self.epsilon - 1.0 / self.epsilon, 0.01)
        return a
        

    def end(self, r: float, extra):
        self.update(r, np.zeros(self.observations[0]))
