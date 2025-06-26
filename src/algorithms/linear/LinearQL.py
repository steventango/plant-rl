from typing import Dict, Tuple  # type: ignore

import numpy as np
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.utils.random import sample

from algorithms.BaseAgent import BaseAgent
from utils.policies import egreedy_probabilities


class LinearQL(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.curr_state = None
        self.curr_action = None
        self.decay_eps_frac = params.get(
            "decay_eps_frac", False
        )  # Fraction of total env steps at which we stop decaying epsilon, if false then there is no decay

        # create initial weights, set to 0
        self.w = np.zeros((actions, self.observations[0]), dtype=np.float64)

    def get_egreedy_action(self, s):
        qs = np.dot(self.w, s)
        probs = egreedy_probabilities(qs, self.actions, self.epsilon)
        a = sample(probs, rng=self.rng)
        return a

    def update(self, r, s_next):
        next_q = np.dot(self.w, s_next).max()
        delta = (
            r + self.gamma * next_q - np.dot(self.w[self.curr_action], self.curr_state)  # type: ignore
        )
        self.w[self.curr_action] += self.alpha * delta * self.curr_state

    def start(self, s: np.ndarray):  # type: ignore
        a = self.get_egreedy_action(s)
        self.curr_state = s
        self.curr_action = a
        return a

    def step(self, r: float, s_next, extra):  # type: ignore
        self.update(r, s_next)
        a = self.get_egreedy_action(s_next)
        self.curr_action = a
        self.curr_state = s_next
        # Decay epsilon
        if self.decay_eps_frac:
            self.epsilon = max(
                self.epsilon - 0.1 / (self.params["exp_len"] * self.decay_eps_frac), 0
            )  # Problem needs to have this param set in main using stride info from env

        return a

    def end(self, r: float, extra):  # type: ignore
        self.update(r, np.zeros(self.observations[0]))
