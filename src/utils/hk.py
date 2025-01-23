import jax
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

from typing import Callable, Dict, Optional, Sequence

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