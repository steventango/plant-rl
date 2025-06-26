import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import curve_percentile_bootstrap_ci, extract_learning_curves
from RlEvaluation.utils.pandas import split_over_column

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference("jmlr")

COLORS = {
    "QL": "red",
    "constant": "black",
    "ESARSA": "blue",
}


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=["alpha", "n_step"],
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
        for stride, stride_df in split_over_column(env_df, col="environment.stride"):  # type: ignore
            f_all, ax_all = plt.subplots()

            for alg, alg_df in split_over_column(stride_df, col="algorithm"):  # type: ignore
                print(alg)

                hyper2metric = {}
                for alpha in alg_df["alpha"].unique():  # type: ignore
                    for n_step in alg_df["n_step"].unique():  # type: ignore
                        xs, ys = extract_learning_curves(
                            alg_df,  # type: ignore
                            (alpha, n_step),
                            metric="return",
                            interpolation=None,  # type: ignore
                        )
                        metric = [r[-1] for r in ys]
                        hyper2metric[(alpha, n_step)] = np.mean(metric)

                best_hyper = max(hyper2metric, key=hyper2metric.get)  # type: ignore
                print(f"Best hypers for {alg} = {best_hyper}")

                # Reward Plot
                xs, ys = extract_learning_curves(
                    alg_df,  # type: ignore
                    best_hyper,
                    metric="return",
                    interpolation=None,  # type: ignore
                )
                xs = np.asarray(xs)
                ys = np.asarray(ys)

                res_r = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                )

                """
                f, ax = plt.subplots(2, 1, figsize=(10, 8))

                ax[0].plot(rescale_time(xs[0], stride), res_r.sample_stat, color=COLORS[alg], linewidth=0.5, label='Reward')
                ax[0].fill_between(rescale_time(xs[0], stride), res_r.ci[0], res_r.ci[1], color=COLORS[alg], alpha=0.2)
                ax[0].set_xlim(0, 168)
                ax[0].set_ylabel('Accumulated Reward')
                ax[0].set_title(f'{alg} Learning Curves (stride={stride})')
                ax[0].legend()

                save(
                    save_path=f'{path}/plots',
                    plot_name=f'{alg}_reward_stride={stride}_{best_hyper}'
                )

                # Action Plot with All Individual Lines
                xs_a, ys_a = extract_learning_curves(alg_df, best_hyper, metric='action', interpolation=None)
                xs_a = np.asarray(xs_a)
                ys_a = np.asarray(ys_a)

                res_a = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys_a,
                    statistic=Statistic.mean,
                )

                ax[1].plot(rescale_time(xs_a[0], stride), res_a.sample_stat, color=COLORS[alg], linewidth=0.8, label='Mean Action')
                ax[1].fill_between(rescale_time(xs_a[0], stride), res_a.ci[0], res_a.ci[1], color=COLORS[alg], alpha=0.2)
                ax[1].set_xlim(0, 168)
                ax[1].set_ylabel('Action')
                ax[1].set_xlabel('Day Time [Hours]')
                ax[1].legend()

                save(
                    save_path=f'{path}/plots',
                    plot_name=f'{alg}_action_all_lines_stride={stride}_{best_hyper}'
                )

                plt.close(f)
                """

                f, ax = plt.subplots(figsize=(10, 8))

                # Action Plot
                xs_a, ys_a = extract_learning_curves(
                    alg_df,  # type: ignore
                    best_hyper,
                    metric="action",
                    interpolation=None,  # type: ignore
                )
                xs_a = np.asarray(xs_a)
                ys_a = np.asarray(ys_a)

                res_a = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys_a,
                    statistic=Statistic.mean,
                )

                ax.plot(
                    rescale_time(xs_a[0], stride),
                    res_a.sample_stat,
                    color=COLORS[alg],
                    linewidth=0.8,
                    label="Mean Action",
                )
                ax.fill_between(
                    rescale_time(xs_a[0], stride),
                    res_a.ci[0],
                    res_a.ci[1],
                    color=COLORS[alg],
                    alpha=0.2,
                )
                ax.set_xlim(0, 168)
                ax.set_ylabel("Action")
                ax.set_xlabel("Day Time [Hours]")
                ax.legend()

                save(
                    save_path=f"{path}/plots",
                    plot_name=f"{alg}_action_all_lines_stride={stride}_{best_hyper}",
                )

                plt.close(f)

                ax_all.plot(
                    rescale_time(xs[0], stride),
                    res_r.sample_stat,
                    label=f"{alg}",
                    color=COLORS[alg],
                    linewidth=0.5,
                )
                ax_all.fill_between(
                    rescale_time(xs[0], stride),
                    res_r.ci[0],
                    res_r.ci[1],
                    color=COLORS[alg],
                    alpha=0.2,
                )

            ax_all.set_xlim(0, 168)
            ax_all.set_title(f"All Algorithms (stride={stride})")
            ax_all.set_ylabel("Accumulated Reward")
            ax_all.legend()

            save(save_path=f"{path}/plots", plot_name=f"all_algos_stride={stride}")

            plt.close(f_all)


def rescale_time(x, stride):
    base_step = 10 / 60  # spreadsheet time step is 10 minutes
    return x * base_step * stride  # x-values in units of hours


def rescale_return(y, min, max):
    return (y - min) / (max - min)


if __name__ == "__main__":
    main()
