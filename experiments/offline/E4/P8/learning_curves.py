import os
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

total_days = 12

THIS_AGENT = "ESARSA0_TOD"


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
        for alg, alg_df in split_over_column(env_df, col="algorithm"):
            if alg == THIS_AGENT:
                sub_df = alg_df
                # sub_df = alg_df[(alg_df['w0']==W0) & (alg_df['epsilon']==EP)]
                report = Hypers.select_best_hypers(
                    sub_df,
                    metric="reward",
                    prefer=Hypers.Preference.high,
                    time_summary=TimeSummary.last_n_percent_sum,
                    statistic=Statistic.mean,
                )

                print("-" * 25)
                print(env, alg)
                Hypers.pretty_print(report)

                xs, ys = extract_learning_curves(
                    sub_df,
                    report.best_configuration,
                    metric="return",
                    interpolation=None,
                )
                xs = np.asarray(xs)
                ys = np.asarray(ys)

                f, ax = plt.subplots(5, 1)
                for i in range(5):
                    ax[i].plot(xs[0], ys[i], "g.", label=f"seed{i + 1}", markersize=2)
                    ax[i].set_ylabel("Return")
                    ax[i].set_xlim(0, 68 * 15)

                ax[0].set_title("Return")
                ax[4].set_xlabel("Day Time Steps")

                save(save_path=f"{path}/plots", plot_name=f"{alg}", save_type="jpg")


if __name__ == "__main__":
    main()
