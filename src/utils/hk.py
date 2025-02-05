import jax
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

from typing import Callable, Dict, Optional, Sequence
Init = hk.initializers.Initializer
Layer = Callable[[jax.Array, hk.LSTMState], tuple[jax.Array, hk.LSTMState]]


@cxu.dataclass
class AccumulatedOutput:
    activations: Dict[str, jax.Array]
    out: jax.Array
    state: hk.LSTMState

def accumulatingSequence(fs: Sequence[Layer]):
    def _inner(x: jax.Array, s: hk.LSTMState):
        out: Dict[str, jax.Array] = {}

        y = x
        for f in fs:
            y, s = f(y, s)
            if isinstance(f, hk.Module):
                out[f.name] = y

        return AccumulatedOutput(activations=out, out=y, state=s)
    return _inner
