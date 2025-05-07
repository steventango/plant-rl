from typing import Any, Callable, Sequence

import numpy as np
from numba import njit
from PyExpUtils.utils.arrays import argsmax
from PyExpUtils.utils.random import sample
from PyExpUtils.utils.types import NpList


class Policy:
    def __init__(self, probs: Callable[[Any], NpList], rng: np.random.Generator):
        self.probs = probs
        self.random = rng

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(np.asarray(action_probabilities), rng=self.random)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: Sequence[NpList], rng: np.random.Generator):
    return Policy(lambda s: probs[s], rng)

def fromActionArray(probs: NpList, rng: np.random.Generator):
    return Policy(lambda s: probs, rng)

def createEGreedy(get_values: Callable[[Any], np.ndarray], actions: int, epsilon: float, rng: np.random.Generator):
    probs = lambda state: egreedy_probabilities(get_values(state), actions, epsilon)

    return Policy(probs, rng)

@njit(cache=True)
def egreedy_probabilities(qs: np.ndarray, actions: int, epsilon: float):
    # compute the greedy policy
    max_acts = argsmax(qs)
    pi: np.ndarray = np.zeros(actions)
    for a in max_acts:
        pi[a] = 1. / len(max_acts)

    # compute a uniform random policy
    uniform: np.ndarray = np.ones(actions) / actions

    # epsilon greedy is a mixture of greedy + uniform random
    return (1. - epsilon) * pi + epsilon * uniform


@njit(cache=True)
def boltzmann_probabilities(qs: np.ndarray, actions: int, temperature: float):
    if temperature <= 0:
        # Fallback to greedy if temperature is zero or negative to avoid division by zero or unexpected behavior.
        # Or, raise an error, or handle as per specific requirements.
        # Here, we choose greedy for simplicity.
        max_acts = argsmax(qs)
        pi: np.ndarray = np.zeros(actions)
        for a in max_acts:
            pi[a] = 1.0 / len(max_acts)
        return pi

    # Subtract max(qs) for numerical stability to prevent overflow with exp
    stable_qs = qs - np.max(qs)
    exp_values = np.exp(stable_qs / temperature)
    sum_exp_values = np.sum(exp_values)
    if sum_exp_values == 0:
        # If all exp_values are zero (e.g., due to underflow with very negative qs and high temperature),
        # fallback to a uniform distribution.
        return np.ones(actions) / actions

    return exp_values / sum_exp_values


def createBoltzmann(
    get_values: Callable[[Any], np.ndarray], actions: int, temperature: float, rng: np.random.Generator
):
    probs = lambda state: boltzmann_probabilities(get_values(state), actions, temperature)
    return Policy(probs, rng)
