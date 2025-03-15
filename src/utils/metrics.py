import typing as tp

import jax
import jax.numpy as jnp


class UnbiasedExponentialMovingAverage:
    """Unbiased Exponential Moving Average.

    Reference: Sutton & Barto (2018) Exercise 2.7

    Example usage::

      >>> import jax.numpy as jnp
      >>> from flax import nnx

      >>> batch_loss = jnp.array([1, 2, 3, 4])
      >>> batch_loss2 = jnp.array([3, 2, 1, 0])

      >>> metrics = nnx.metrics.UnbiasedExponentialMovingAverage()
      >>> metrics.compute()
      Array(nan, dtype=float32)
      >>> metrics.update(values=batch_loss)
      >>> metrics.compute()
      Array(2.501251, dtype=float32)
      >>> metrics.update(values=batch_loss2)
      >>> metrics.compute()
      Array(1.998997, dtype=float32)
      >>> metrics.reset()
      >>> metrics.compute()
      Array(nan, dtype=float32)
    """

    def __init__(self, shape=1, alpha: float = 0.001):
        """Pass in a string denoting the key-word argument that :func:`update` will use to derive the new value.
        For example, constructing the metric as ``uema = UnbiasedExponentialMovingAverage('test')`` would allow you to make updates with
        ``uema.update(test=new_value)``.

        Args:
          shape: int or sequence of ints specifying the shape of the created array.
          alpha: the smoothing factor. Defaults to ``0.001``.
        """
        self.alpha = alpha
        self.shape = shape
        self.reset()

    def reset(self) -> None:
        """Reset this ``UnbiasedExponentialMovingAverage``."""
        self.total = jnp.zeros(self.shape, dtype=jnp.float32)
        self.count_trace = jnp.array(0, dtype=jnp.int32)

    def update(self, values: tp.Union[int, float, jax.Array]) -> None:
        """In-place update this ``UnbiasedExponentialMovingAverage``. This
        method will use ``values`` to update the metric.

        Args:
            values: the values we want to use to update this metric.
        """
        values = jnp.atleast_1d(values).astype(jnp.float32)
        for value in values:
            self.count_trace += self.alpha * (1 - self.count_trace)
            beta = self.alpha / self.count_trace
            self.total = (1 - beta) * self.total + beta * value

    def compute(self) -> jax.Array:
        """Compute and return the unbiased exponential moving average."""
        return self.total if self.count_trace > 0 else jnp.full(self.shape, jnp.nan, dtype=jnp.float32)
