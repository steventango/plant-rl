import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")


import enum

import matplotlib.pyplot as plt
import numpy as np
import RlEvaluation.hypers as Hypers
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import extract_learning_curves
from RlEvaluation.utils.pandas import split_over_column

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs
from utils.metrics import UnbiasedExponentialMovingAverage as UEMA

setDefaultConference("neurips")

total_days = 12


def last_n_percent_sum(x, n=0.2):
    return np.nansum(x[:, int((1 - n) * x.shape[1]) :], axis=1)


class TimeSummary(enum.Enum):
    last_n_percent_sum = enum.member(last_n_percent_sum)


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col="environment",
        algorithm_col="algorithm",
        make_global=True,
    )

    df = results.combine(
        folder_columns=(None, None, None, "environment"),
        file_col="algorithm",
    )

    assert df is not None

    results.get_any_exp()

    for env, env_df in split_over_column(df, col="environment"):
        f, ax = plt.subplots(1)
        for alg, sub_df in split_over_column(env_df, col="algorithm"):  # type: ignore
            report = Hypers.select_best_hypers(
                sub_df,  # type: ignore
                metric="reward",
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.last_n_percent_sum,  # type: ignore
                statistic=Statistic.mean,
            )

            print("-" * 25)
            print(env, alg)
            Hypers.pretty_print(report)

            xs, ys = extract_learning_curves(
                sub_df,
                report.best_configuration,
                metric="action",
                interpolation=None,  # type: ignore
            )
            xs = np.asarray(xs)
            ys = np.asarray(ys)

            ema_action = calculate_ema(xs, ys)
            mean_ema_action = np.mean(ema_action, axis=0)
            line = ax.plot(xs[0], mean_ema_action, linewidth=1, label=alg)
            color = line[0].get_color()
            for i in range(5):
                ax.plot(xs[0], ema_action[i], linewidth=0.5, alpha=0.5, color=color)

        ax.axhline(y=0.95, color="k", linestyle="--", label="0.95")
        ax.set_title("Action")
        ax.set_xlabel("Day Time Steps")
        ax.set_ylabel("Action EMA")
        ax.legend()

        save(save_path=f"{path}/plots", plot_name="action", save_type="jpg")


def calculate_ema(xs, ys):
    num_trajectories = xs.shape[0]
    emas = [UEMA() for _ in range(num_trajectories)]
    ema_action = [[] for _ in range(num_trajectories)]
    for i in range(num_trajectories):
        for j in range(len(xs[0])):
            emas[i].update(ys[i][j])
            ema_action[i].append(emas[i].compute().item())

    ema_action = np.asarray(ema_action)
    return ema_action


if __name__ == "__main__":
    main()
