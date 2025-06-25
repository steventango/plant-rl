from typing import Callable

import numpy as np
from PyExpUtils.collection.Sampler import Sampler


class WindowAverage(Sampler):
    def __init__(self, size: int):
        self.size = size
        self.window = np.zeros(self.size)
        self.n_inserts = 0

    def next(self, v: float):
        self.window[self.n_inserts % self.size] = v
        self.n_inserts += 1
        return self.window[: min(self.n_inserts, self.size)].mean()

    def next_eval(self, c: Callable[[], float]):
        v = c()
        return self.next(v)

    def repeat(self, v: float, times: int):
        for _ in range(times):
            self.next(v)

    def end(self):
        return None
