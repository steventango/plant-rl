import os
import sys

sys.path.append(os.getcwd() + "/src")
import enum
from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RlEvaluation.backend.statistics as bs
import RlEvaluation.backend.temporal as bt
import RlEvaluation.hypers as Hypers
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import DataDefinition, data_definition, maybe_global
from RlEvaluation.interpolation import Interpolation
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import (
    TimeSummary,
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)
from RlEvaluation.utils.pandas import split_over_column, subset_df

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference("neurips")

COLORS = {"tc-ESARSA": "blue"}


def maybe_convert_to_array(x):
    if isinstance(x, float):
        return x
    x = eval(x)
    if isinstance(x, bytes):
        return np.frombuffer(x)
    return x


def extract_learning_curves(
    df: pd.DataFrame,
    hyper_vals: Tuple[Any, ...],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    dd = maybe_global(data_definition)
    cols = set(dd.hyper_cols).intersection(df.columns)
    sub = subset_df(df, list(cols), hyper_vals)

    groups = sub.groupby(dd.seed_col, dropna=False)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _, group in groups:
        non_na = group[group[metric].notna()]
        x = non_na[dd.time_col].to_numpy()
        y = non_na[metric].to_numpy()

        idx = np.argwhere(x[1:] <= x[:-1])

        if idx.size > 0:
            x = x[idx[-1][0] + 1:]
            y = y[idx[-1][0] + 1:]

        if interpolation is not None:
            x, y = interpolation(x, y)

        xs.append(x)
        ys.append(y)

    xs = np.stack(xs)
    ys = np.stack(ys)

    return xs, ys


def main():
    path, should_save, save_type = parseCmdLineArgs()

    data_definition(
        hyper_cols=[],
        seed_col='seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',
        make_global=True,
    )

    df = pd.read_csv(f"{path}/data.csv")

    for metric in ["state", "action", "reward"]:
        df[metric] = df[metric].apply(maybe_convert_to_array)
        for alg, sub_df in split_over_column(df, col="algorithm"):
            print("-" * 25)
            print(alg)

            xs, ys = extract_learning_curves(sub_df, tuple(), metric=metric, interpolation=None)
            x = xs[0]
            y = ys[0]
            y = np.stack(y)
            if y.ndim == 1:
                y = y[:, np.newaxis]
            f, axs = plt.subplots(y.shape[1], 1, squeeze=False, sharex=True)
            axs = axs.flatten()
            total_days = int(np.max(x) * 5 / 60 / 24)
            for j, (ax, yj) in enumerate(zip(axs, y.T)):
                ax.step(rescale_time(x, 1), yj, label=f"{alg}")
                ax.set_ylabel(metric.capitalize() + f"[{j}]")
                for k in range(total_days + 1):
                    ax.axvline(x=12 * k, color="k", linestyle="--", linewidth=0.5)
            axs[0].set_title(f"{metric.capitalize()}")
            axs[-1].set_xlabel("Time [Hours]")

            save(save_path=f"{path}/plots", plot_name=f"{alg}_{metric}", save_type="jpg")


def rescale_time(x, stride):
    base_step = 5 / 60
    return x * base_step / stride


if __name__ == "__main__":
    main()
