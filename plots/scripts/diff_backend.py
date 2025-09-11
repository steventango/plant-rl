import numpy as np

from numba import prange
from typing import Any, Tuple, NamedTuple

import diff_stats as bs
import RlEvaluation._utils.numba as nbu


# -----------------------------
# -- Statistical Simulations --
# -----------------------------

@nbu.njit(parallel=True)
def curve_percentile_bootstrap_ci_diff(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    statistic: Any,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    n_measurements = y.shape[1]

    lo = np.empty(n_measurements, dtype=np.float64)
    center = np.empty(n_measurements, dtype=np.float64)
    hi = np.empty(n_measurements, dtype=np.float64)

    for i in prange(n_measurements):
        res = bs.percentile_bootstrap_ci_diff(
            rng,
            x[:, i],
            y[:, i],
            statistic=statistic,
            alpha=alpha,
            iterations=iterations,
        )

        lo[i] = res.ci[0]
        center[i] = res.sample_stat
        hi[i] = res.ci[1]

    return CurvePercentileBootstrapResult(
        sample_stat=center,
        ci=(lo, hi),
    )


class CurvePercentileBootstrapResult(NamedTuple):
    sample_stat: np.ndarray
    ci: Tuple[np.ndarray, np.ndarray]
