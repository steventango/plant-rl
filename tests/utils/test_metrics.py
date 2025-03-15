import jax.numpy as jnp
import pytest

from src.metrics import (
    NMSE,
    UnbiasedExponentialMovingAverage,
    UnbiasedExponentialMovingNMSE,
    UnbiasedExponentialMovingWelford,
)


class TestUnbiasedExponentialMovingAverage:
    def test___init__(self):
        metric = UnbiasedExponentialMovingAverage()
        assert jnp.isnan(metric.compute())

    def test_reset(self):
        metric = UnbiasedExponentialMovingAverage()
        metric.update(values=jnp.array([1, 2, 3, 4]))
        metric.reset()
        assert jnp.isnan(metric.compute())

    def test_update(self):
        metric = UnbiasedExponentialMovingAverage()
        metric.update(values=jnp.array([1, 2, 3, 4]))

    def test_compute(self):
        metric = UnbiasedExponentialMovingAverage()
        metric.update(values=1)
        uema = metric.compute()
        assert uema == pytest.approx(1)
        metric.update(values=2)
        uema = metric.compute()
        assert uema == pytest.approx(1.500250)
        metric.update(values=3)
        uema = metric.compute()
        assert uema == pytest.approx(2.000667)
        metric.update(values=4)
        uema = metric.compute()
        assert uema == pytest.approx(2.501251)
        metric.update(values=jnp.array([3, 2, 1, 0]))
        uema = metric.compute()
        assert uema == pytest.approx(1.998997)
