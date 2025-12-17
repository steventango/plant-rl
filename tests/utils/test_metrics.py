import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from utils.metrics import UnbiasedExponentialMovingAverage, iqm


class TestUnbiasedExponentialMovingAverage:
    def test___init__(self):
        uema = UnbiasedExponentialMovingAverage()
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert jnp.equal(metric, 0)

    def test_reset(self):
        uema = UnbiasedExponentialMovingAverage()
        uema.update(values=jnp.array([1, 2, 3, 4]))
        uema.reset()
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (1,)
        assert jnp.equal(metric, 0)

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

    def test__init__with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(jnp.equal(metric, 0))

    def test_reset_with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        uema.update(values=jnp.array([[1, 2], [3, 4]]))
        uema.reset()
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(jnp.equal(metric, 0))

    def test_update_with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        uema.update(values=jnp.array([[1, 2], [3, 4]]))

    def test_compute_with_shape(self):
        uema = UnbiasedExponentialMovingAverage(shape=2)
        uema.update(values=jnp.array([1, 1]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(1))
        uema.update(values=jnp.array([2, 2]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(1.500250))
        uema.update(values=jnp.array([3, 3]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(2.000667))
        uema.update(values=jnp.array([4, 4]))
        metric = uema.compute()
        assert metric.dtype == jnp.float32
        assert metric.shape == (2,)
        assert jnp.all(metric == pytest.approx(2.501251))

    def test_plot_beta(self):
        uema = UnbiasedExponentialMovingAverage(shape=1)
        beta_history = []
        count_trace_history = []
        for value in np.arange(10000):
            uema.count_trace += uema.alpha * (1 - uema.count_trace)
            count_trace_history.append(uema.count_trace)
            beta = uema.alpha / uema.count_trace
            beta_history.append(beta)
            uema.total = (1 - beta) * uema.total + beta * value
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(count_trace_history)
        ax[0].set_xlabel("count_trace")
        ax[1].plot(beta_history, label=f"last beta={beta_history[-1]:.2g}")
        ax[1].legend()
        ax[1].set_xlabel("beta")
        plt.savefig("tests/utils/plot_trace.jpg")


def test_iqm():
    data = jnp.array([-1000, 1, 2, 3, 4, 5, 1000])
    metric = iqm(data, 0.05)
    assert metric == pytest.approx(3)
