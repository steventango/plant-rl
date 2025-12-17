from typing import Callable

from ml_instrumentation.utils import Pipe
from ml_instrumentation.Sampler import Sampler


class Last(Pipe):
    def __init__(self, *args: Sampler):
        super().__init__(*args)
        self.last: float | None = None

    def next(self, v: float) -> float | None:
        self.last = super().next(v)

    def next_eval(self, c: Callable[[], float]) -> float | None:
        self.last = super().next_eval(c)

    def end(self):
        return self.last
