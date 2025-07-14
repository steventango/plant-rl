# TODO: steady-state performance
#  - Fit a model (piecewise linear with one node?) and report bias unit for second part

import enum
import numpy as np
import pandas as pd

from typing import Any, List, Sequence, Tuple
from RlEvaluation.config import DataDefinition, maybe_global
from RlEvaluation.interpolation import Interpolation
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import subset_df

import diff_backend as bt


def curve_percentile_bootstrap_ci_diff(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    statistic: Statistic = Statistic.mean,
    alpha: float = 0.05,
    iterations: int = 10000,
):
    f: Any = statistic.value
    assert x.shape == y.shape, "x and y must have the same shape for curve percentile bootstrap CI diff"
    return bt.curve_percentile_bootstrap_ci_diff(
        rng=rng,
        x=x,
        y=y,
        statistic=f,
        alpha=alpha,
        iterations=iterations,
    )