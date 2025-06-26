import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")


import matplotlib.pyplot as plt
import numpy as np
import RlEvaluation.hypers as Hypers
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import TimeSummary, extract_learning_curves
from RlEvaluation.utils.pandas import split_over_column

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs
from utils.metrics import UnbiasedExponentialMovingAverage as UEMA

setDefaultConference("neurips")

total_days = 12


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
        for replay_ratio, sub_df in sorted(
            split_over_column(env_df, col="replay_ratio"),
            key=lambda x: x[0],  # type: ignore
        ):
            report = Hypers.select_best_hypers(
                sub_df,  # type: ignore
                metric="reward",
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.mean,
                statistic=Statistic.mean,
            )

            print("-" * 25)
            print(env, replay_ratio)
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
            line = ax.plot(
                xs[0],
                mean_ema_action,
                linewidth=1,
                label=f"k={replay_ratio}",
                alpha=0.8,
            )
            color = line[0].get_color()
            for i in range(xs.shape[0]):
                ax.plot(xs[0], ema_action[i], linewidth=0.5, alpha=0.5, color=color)

        ax.axhline(y=0.95, color="k", linestyle="--", label="0.95")
        ax.set_title("Action")
        ax.set_xlabel("Day Time Steps")
        ax.set_ylabel("Action EMA")
        ax.set_ylim(0.5, 1)
        ax.legend()

        save(
            save_path=f"{path}/plots",
            plot_name="action",
            save_type="jpg",
            width=3,
            height_ratio=1 / 3,
        )


def calculate_ema(xs, ys):
    emas = [UEMA() for _ in range(xs.shape[0])]
    ema_action = [[] for _ in range(xs.shape[0])]
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            emas[i].update(ys[i][j])
            ema_action[i].append(emas[i].compute().item())

    ema_action = np.asarray(ema_action)
    return ema_action


if __name__ == "__main__":
    main()
