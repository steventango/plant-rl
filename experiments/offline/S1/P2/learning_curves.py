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
from RlEvaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)
from RlEvaluation.utils.pandas import split_over_column

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference("neurips")

COLORS = {"linear-ESARSA": "blue"}

total_days = 14


def last_20_percent_sum(x):
    return np.nansum(x[:, int(0.8 * x.shape[1]) :], axis=1)


class TimeSummary(enum.Enum):
    last_20_percent_sum = enum.member(last_20_percent_sum)


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
        for _last_day, filter_df in split_over_column(
            env_df,
            col="environment.last_day",  # type: ignore
        ):
            for metric in ["reward", "action"]:
                f, ax = plt.subplots(5, 1)
                for alg, sub_df in split_over_column(filter_df, col="algorithm"):  # type: ignore
                    report = Hypers.select_best_hypers(
                        sub_df,  # type: ignore
                        metric="return",
                        prefer=Hypers.Preference.high,
                        time_summary=TimeSummary.last_20_percent_sum,  # type: ignore
                        statistic=Statistic.mean,
                    )

                    print("-" * 25)
                    print(env, alg)
                    Hypers.pretty_print(report)

                    xs_a, ys_a = extract_learning_curves(
                        sub_df,  # type: ignore
                        report.best_configuration,
                        metric=metric,
                        interpolation=None,
                    )
                    min_length = min([len(xs) for xs in xs_a])
                    xs_a = [xs[:min_length] for xs in xs_a]
                    ys_a = [ys[:min_length] for ys in ys_a]
                    xs_a = np.asarray(xs_a)
                    ys_a = np.asarray(ys_a)

                    curve_percentile_bootstrap_ci(
                        rng=np.random.default_rng(0),
                        y=ys_a,
                        statistic=Statistic.mean,
                    )

                    for i in range(min(5, len(ys_a))):
                        ax[i].plot(
                            rescale_time(xs_a[0], 1),
                            ys_a[i],
                            marker=".",
                            linestyle="",
                            label=f"{alg}",
                            markersize=0.5,
                        )
                        ax[i].set_ylabel(metric.capitalize())
                        for j in range(total_days + 1):
                            ax[i].axvline(
                                x=12 * j, color="k", linestyle="--", linewidth=0.5
                            )
                    ax[0].set_title(
                        f"{metric.capitalize()} Learning Curves over {total_days} Days"
                    )
                    ax[0].legend()
                    ax[-1].set_xlabel("Day Time [Hours]")

                save(save_path=f"{path}/plots", plot_name=metric, save_type="jpg")


def rescale_time(x, stride):
    base_step = 10 / 60  # spreadsheet time step is 10 minutes
    return x * base_step * stride  # x-values in units of hours


if __name__ == "__main__":
    main()
