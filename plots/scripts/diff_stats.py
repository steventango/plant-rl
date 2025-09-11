import numpy as np
import numba as nb
import RlEvaluation._utils.numba as nbu

from scipy.stats import binom
from typing import Any, Callable, List, NamedTuple, Tuple

# ----------------------
# -- Basic Statistics --
# ----------------------

@nbu.njit(inline='always')
def mean(a: np.ndarray, axis: int = 0):
    return np.sum(a, axis=axis) / a.shape[axis]

@nbu.njit(inline='always')
def agg(a: np.ndarray, axis: int = 0):
    return np.sum(a, axis=axis)


# -----------------------------
# -- Statistical Simulations --
# -----------------------------

@nbu.njit
def percentile_bootstrap_ci_diff(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray], Any] = mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    bs = np.empty(iterations, dtype=np.float64)

    for i in range(iterations):
        # Resample each group independently
        x_idxs = rng.integers(0, len(x), size=len(x))
        y_idxs = rng.integers(0, len(y), size=len(y))
        
        # Compute difference of means
        x_bootstrap_mean = statistic(x[x_idxs])
        y_bootstrap_mean = statistic(y[y_idxs])
        bs[i] = x_bootstrap_mean - y_bootstrap_mean

    sample_stat = statistic(x) - statistic(y)

    lo_b = (alpha / 2)
    hi_b = 1 - (alpha / 2)
    lo, hi = np.percentile(bs, (100 * lo_b, 100 * hi_b))

    return PercentileBootstrapResult(
        sample_stat=sample_stat,
        ci=(lo, hi),
    )

class PercentileBootstrapResult(NamedTuple):
    sample_stat: float
    ci: Tuple[float, float]