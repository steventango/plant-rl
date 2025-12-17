from typing import Callable

from ml_instrumentation.Sampler import Sampler


class Mean(Sampler):
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def next(self, v: float):
        self.sum += v
        self.count += 1
        return self.sum / self.count

    def next_eval(self, c: Callable[[], float]):
        v = c()
        return self.next(v)

    def end(self):
        return None
