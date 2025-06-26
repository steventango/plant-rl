import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyExpPlotting.matplot import save, setDefaultConference
from RlEvaluation.config import DataDefinition, data_definition, maybe_global
from RlEvaluation.interpolation import Interpolation, compute_step_return
from RlEvaluation.utils.pandas import split_over_column, subset_df

from experiment.tools import parseCmdLineArgs
from utils.metrics import UnbiasedExponentialMovingAverage as uema
from utils.metrics import iqm

setDefaultConference("neurips")

COLORS = {"tc-ESARSA": "blue"}


def to_numpy(string_list):
    """Converts a string representation of a list of numbers to a NumPy array."""
    if not isinstance(string_list, str):
        return string_list
    try:
        # Remove the brackets and split by space
        numbers_str = string_list.strip("[]").split()
        # Convert the strings to floats and create a NumPy array
        return np.array([float(num) for num in numbers_str])
    except AttributeError:
        return np.nan  # Or handle non-string elements as needed


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
        non_na = group[group[metric].notna()]  # type: ignore
        x = non_na[dd.time_col].to_numpy()  # type: ignore
        y = non_na[metric].to_numpy()  # type: ignore

        idx = np.argwhere(x[1:] <= x[:-1])

        if idx.size > 0:
            x = x[idx[-1][0] + 1 :]
            y = y[idx[-1][0] + 1 :]

        if interpolation is not None:
            x, y = interpolation(x, y)

        xs.append(x)
        ys.append(y)

    xs = np.stack(xs)  # type: ignore
    ys = np.stack(ys)  # type: ignore

    return xs, ys


def main():
    path, should_save, save_type = parseCmdLineArgs()

    data_definition(
        hyper_cols=[],
        seed_col="seed",
        time_col="frame",
        environment_col="environment",
        algorithm_col="algorithm",
        make_global=True,
    )

    df = pd.read_csv(f"{path}/data.csv")

    for metric in ["area", "state", "action", "reward"]:
        df[metric] = df[metric].apply(to_numpy)  # type: ignore
        for alg, sub_df in split_over_column(df, col="algorithm"):
            print("-" * 25)
            print(metric, alg)

            xs, ys = extract_learning_curves(
                sub_df,  # type: ignore
                tuple(),
                metric=metric,
                interpolation=None,  # type: ignore
            )
            x = xs[0]
            y = ys[0]
            y = np.stack(y)  # type: ignore
            if y.ndim == 1:
                y = y[:, np.newaxis]
            m = y.shape[1]
            rows = m
            if metric == "area":
                rows += 1
            f, axs = plt.subplots(rows, 1, squeeze=False, sharex=True)
            axs = axs.flatten()
            total_days = int(np.max(x) * 5 / 60 / 24)
            for j, (ax, yj) in enumerate(zip(axs, y.T, strict=False)):
                x_plot = rescale_time(x, 1)
                # Draw horizontal segments without vertical connectors
                for i in range(len(x_plot) - 1):
                    ax.hlines(
                        y=yj[i],
                        xmin=x_plot[i],
                        xmax=x_plot[i + 1],
                        color="C0",
                        label=f"{alg}" if i == 0 else None,
                    )
                ax.set_ylabel(metric + f"[{j}]" if m > 1 else metric)
                for k in range(total_days + 1):
                    ax.axvline(x=24 * k, color="k", linestyle="--", linewidth=0.5)
                if metric in {"area", "reward"}:
                    u = uema(alpha=0.1)
                    stat = []
                    for yj_i in yj:
                        u.update(yj_i)
                        stat.append(u.compute())
                    stat = np.array(stat)
                    ax.plot(x_plot, stat, color="C1", label="UEMA")
                if metric == "reward":
                    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                    ax.set_ylim(bottom=-0.1)
                    _, returns = compute_step_return(x, yj, len(x))
                    total_return = np.sum(returns)
                    ax.set_ylim([0, 1])
                    print(f"Return: {total_return:.2f}")
            if metric == "area":
                # plot IQM of the area
                x_plot = rescale_time(x, 1)
                for i in range(len(x_plot) - 1):
                    axs[-1].hlines(
                        y=iqm(y.T[:, i], 0.1),
                        xmin=x_plot[i],
                        xmax=x_plot[i + 1],
                        color="C0",
                        label=f"{alg}" if i == 0 else None,
                    )
                axs[-1].set_ylabel("IQM")
                for k in range(total_days + 1):
                    axs[-1].axvline(x=24 * k, color="k", linestyle="--", linewidth=0.5)
                u = uema(alpha=0.1)
                stat = []
                for y_i in y:
                    u.update(iqm(y_i, 0.1))
                    stat.append(u.compute())
                stat = np.array(stat)
                axs[-1].plot(x_plot, stat, color="C1", label="UEMA")

            axs[0].set_title(f"{metric.capitalize()}")
            axs[-1].set_xlabel("Time [Hours]")

            save(
                save_path=f"{path}/plots",
                plot_name=f"{alg}_{metric}",
                save_type="jpg",
                height_ratio=0.2 * m,
            )


def rescale_time(x, stride):
    base_step = 5 / 60
    return x * base_step / stride


if __name__ == "__main__":
    main()
