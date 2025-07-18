import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")
os.environ["NUMBA_DISABLE_JIT"] = "1"

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

setDefaultConference("neurips")

total_days = 14

W0 = 0.0
EP = 0.1


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
        f, ax = plt.subplots(5, 1)
        for alg, alg_df in split_over_column(env_df, col="algorithm"):  # type: ignore
            sub_df = alg_df
            sub_df = alg_df[(alg_df["w0"] == W0) & (alg_df["epsilon"] == EP)]
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

            xs_w0, ys_w0 = extract_learning_curves(
                sub_df,  # type: ignore
                report.best_configuration,
                metric="weight0",
                interpolation=None,  # type: ignore
            )
            xs_w1, ys_w1 = extract_learning_curves(
                sub_df,  # type: ignore
                report.best_configuration,
                metric="weight1",
                interpolation=None,  # type: ignore
            )
            xs_w2, ys_w2 = extract_learning_curves(
                sub_df,  # type: ignore
                report.best_configuration,
                metric="weight2",
                interpolation=None,  # type: ignore
            )
            xs_w3, ys_w3 = extract_learning_curves(
                sub_df,  # type: ignore
                report.best_configuration,
                metric="weight3",
                interpolation=None,  # type: ignore
            )

            for i in range(5):
                ys_w = np.vstack([ys_w3[i], ys_w2[i], ys_w1[i], ys_w0[i]])
                print(np.max(ys_w))
                print(np.min(ys_w))
                ax[i].imshow(
                    ys_w,
                    aspect=40,
                    extent=(0, 1008, 0, 3),
                    vmin=-0.03,
                    vmax=0.03,
                    cmap="Purples",
                )
                ax[i].set_ylabel("Action")
                ax[i].set_xlim([0, 1008])
                for j in range(total_days + 1):
                    ax[i].axvline(x=72 * j, color="k", linestyle="--", linewidth=0.5)

            ax[0].set_title(
                f"Bandit's value function over {total_days} days; w0={W0}, ep={EP}; best_score={report.best_score:.3f}"
            )
            ax[4].set_xlabel("Daytime Steps")

        save(
            save_path=f"{path}/plots",
            plot_name=f"{alg}-w0={W0}-ep={EP}",  # type: ignore
            save_type="jpg",
        )


def rescale_time(x, stride):
    base_step = 10 / 60  # spreadsheet time step is 10 minutes
    return x * base_step * stride  # x-values in units of hours


if __name__ == "__main__":
    main()
