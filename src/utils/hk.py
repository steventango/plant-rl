from typing import Callable, Dict, Sequence

import haiku as hk
import jax

import utils.chex as cxu

Init = hk.initializers.Initializer
Layer = Callable[[jax.Array], jax.Array]


@cxu.dataclass
class AccumulatedOutput:
    activations: Dict[str, jax.Array]
    out: jax.Array


def accumulatingSequence(fs: Sequence[Layer]):
    def _inner(x: jax.Array):
        out: Dict[str, jax.Array] = {}

        y = x
        for f in fs:
            y = f(y)
            if isinstance(f, hk.Module):
                out[f.name] = y

        return AccumulatedOutput(activations=out, out=y)

    return _inner
