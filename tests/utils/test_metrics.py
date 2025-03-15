import jax.numpy as jnp
import pytest

from utils.metrics import UnbiasedExponentialMovingAverage


class TestUnbiasedExponentialMovingAverage:
    def test___init__(self):
        uema = UnbiasedExponentialMovingAverage()
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert jnp.isnan(metric)

    def test_reset(self):
        uema = UnbiasedExponentialMovingAverage()
        uema.update(values=jnp.array([1, 2, 3, 4]))
        uema.reset()
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert jnp.isnan(metric)

    def test_update(self):
        uema = UnbiasedExponentialMovingAverage()
        uema.update(values=jnp.array([1, 2, 3, 4]))

    def test_compute(self):
        uema = UnbiasedExponentialMovingAverage()
        uema.update(values=1)
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert metric == pytest.approx(1)
        uema.update(values=2)
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert metric == pytest.approx(1.500250)
        uema.update(values=3)
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert metric == pytest.approx(2.000667)
        uema.update(values=4)
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert metric == pytest.approx(2.501251)
        uema.update(values=jnp.array([3, 2, 1, 0]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert metric == pytest.approx(1.998997)

    def test__init__with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(jnp.isnan(metric))

    def test_reset_with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        uema.update(values=jnp.array([[1, 2], [3, 4]]))
        uema.reset()
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(jnp.isnan(metric))

    def test_update_with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        uema.update(values=jnp.array([[1, 2], [3, 4]]))

    def test_compute_with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        uema.update(values=jnp.array([[1, 1]]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(1))
        uema.update(values=jnp.array([[2, 2]]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(1.500250))
        uema.update(values=jnp.array([[3, 3]]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(2.000667))
        uema.update(values=jnp.array([[4, 4]]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(2.501251))
        uema.update(values=jnp.array([[3, 3], [2, 2], [1, 1], [0, 0]]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(1.998997))
