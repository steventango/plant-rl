import typing as tp

import jax
import jax.numpy as jnp


class UnbiasedExponentialMovingAverage:
    """Unbiased Exponential Moving Average metric.

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

    def __init__(self, argname: str = "values", alpha: float = 0.001):
        """Pass in a string denoting the key-word argument that :func:`update` will use to derive the new value.
        For example, constructing the metric as ``uema = UnbiasedExponentialMovingAverage('test')`` would allow you to make updates with
        ``uema.update(test=new_value)``.

        Args:
          argname: an optional string denoting the key-word argument that
            :func:`update` will use to derive the new value. Defaults to
            ``'values'``.
        """
        self.argname = argname

        self.alpha = alpha

        self.total = jnp.array(0, dtype=jnp.float32)
        self.count_trace = jnp.array(0, dtype=jnp.int32)

    def reset(self) -> None:
        """Reset this ``Metric``."""
        self.total = jnp.array(0, dtype=jnp.float32)
        self.count_trace = jnp.array(0, dtype=jnp.int32)

    def update(self, **kwargs) -> None:
        """In-place update this ``Metric``. This method will use the value from
        ``kwargs[self.argname]`` to update the metric, where ``self.argname`` is
        defined on construction.

        Args:
          **kwargs: the key-word arguments that contains a ``self.argname``
            entry that maps to the value we want to use to update this metric.
        """
        if self.argname not in kwargs:
            raise TypeError(f"Expected keyword argument '{self.argname}'")
        values: tp.Union[int, float, jax.Array] = kwargs[self.argname]
        if isinstance(values, (int, float)) or values.ndim == 0:
            values = jnp.array([values], dtype=jnp.float32)
        for value in values.flatten():
            self.count_trace += self.alpha * (1 - self.count_trace)
            beta = self.alpha / self.count_trace
            self.total = (1 - beta) * self.total + beta * value

    def compute(self) -> jax.Array:
        """Compute and return the unbiased exponential moving average."""
        return self.total if self.count_trace > 0 else jnp.nan
