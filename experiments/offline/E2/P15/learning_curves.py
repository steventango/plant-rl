import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import curve_percentile_bootstrap_ci, extract_learning_curves
from RlEvaluation.utils.pandas import split_over_column

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference("neurips")
setFonts(
    font_size=8
)  # for a small plot ("PaperDimensions" of neurips has been changed to column_width=3, text_width=3)
# setDefaultConference('jmlr')  # for a large plot

PLOT_THIS = "GAC-sweep-n0"

COLORS = {"GAC-sweep-n0": "red"}

N = {"GAC-sweep-n0": 0}


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=["critic_lr", "actor_lr_scale", "tau"],
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

    for _env, env_df in split_over_column(df, col="environment"):
        for alg, alg_df in split_over_column(env_df, col="algorithm"):  # type: ignore
            if alg == PLOT_THIS:
                f, ax = plt.subplots(2, 1)

                # Pick the best hypers by total reward over 14 days (1 episode)
                hyper2metric = {}
                for clr in alg_df["critic_lr"].unique():  # type: ignore
                    for alr in alg_df["actor_lr_scale"].unique():  # type: ignore
                        for tau in alg_df["tau"].unique():  # type: ignore
                            xs, ys = extract_learning_curves(
                                alg_df,  # type: ignore
                                (clr, alr, tau),
                                metric="return",
                                interpolation=None,
                            )
                            assert len(xs) == 5
                            metric = [r[-1] for r in ys]
                            hyper2metric[(clr, alr, tau)] = np.mean(metric)

                best_hyper = max(hyper2metric, key=hyper2metric.get)  # type: ignore
                print(f"Best hypers for {alg} = {best_hyper}")

                # Plot reward history averaged over 5 seeds
                xs, ys = extract_learning_curves(
                    alg_df,  # type: ignore
                    best_hyper,
                    metric="return",
                    interpolation=None,  # type: ignore
                )
                xs = np.asarray(xs)
                ys = np.asarray(ys)

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                )

                ax[0].plot(
                    rescale_time(xs[0], 1),
                    res.sample_stat,
                    label=f"n_hidden={N[alg]}",
                    color=COLORS[alg],
                    linewidth=0.5,
                )
                ax[0].fill_between(
                    rescale_time(xs[0], 1),
                    res.ci[0],
                    res.ci[1],
                    color=COLORS[alg],
                    alpha=0.2,
                )
                ax[0].legend()
                ax[0].set_title("Learning Curves")
                ax[0].set_ylabel("Accumulated Reward")

                # Plot action history averaged over 5 seeds
                xs_a, ys_a = extract_learning_curves(
                    alg_df,  # type: ignore
                    best_hyper,
                    metric="action",
                    interpolation=None,  # type: ignore
                )
                xs_a = np.asarray(xs_a)
                ys_a = np.asarray(ys_a)

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys_a,
                    statistic=Statistic.mean,
                )

                ax[1].plot(
                    rescale_time(xs_a[0], 1),
                    res.sample_stat,
                    color=COLORS[alg],
                    linewidth=0.5,
                )
                ax[1].fill_between(
                    rescale_time(xs_a[0], 1),
                    res.ci[0],
                    res.ci[1],
                    color=COLORS[alg],
                    alpha=0.2,
                )
                ax[1].set_ylabel("Action")
                ax[1].set_xlabel("Day Time [Hours]")

                save(save_path=f"{path}/plots", plot_name=f"{alg}")


def rescale_time(x, stride):
    base_step = 10 / 60  # spreadsheet time step is 10 minutes
    return x * base_step * stride  # x-values in units of hours


def rescale_return(y, min, max):
    return (y - min) / (max - min)


if __name__ == "__main__":
    main()
