import os
import sys

sys.path.append(os.getcwd() + "/src")

import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

from RlEvaluation.config import data_definition
from RlEvaluation.temporal import extract_learning_curves, curve_percentile_bootstrap_ci
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import split_over_column


# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference("jmlr")

THIS_AGENT = "GAC-sweep"


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=["critic_lr", "actor_lr_scale", "hidden_dim"],
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

    exp = results.get_any_exp()

    for _env, env_df in split_over_column(df, col="environment"):
        for alg, alg_df in split_over_column(env_df, col="algorithm"):
            print(alg)

            # Pick the best learning rate
            lr2metric = {}
            for lr in alg_df["critic_lr"].unique():
                for actor_lr in alg_df["actor_lr_scale"].unique():
                    for hidden_dim in alg_df["hidden_dim"].unique():
                        xs, ys = extract_learning_curves(
                            alg_df,
                            (lr, actor_lr, hidden_dim),
                            metric="return",
                            interpolation=None,
                        )
                        print(len(xs))
                        metric = [auc(t, r) for t, r in zip(xs, ys, strict=False)]
                        lr2metric[(lr, actor_lr, hidden_dim)] = np.mean(metric)

            best_lr = max(lr2metric, key=lr2metric.get)
            xs, ys = extract_learning_curves(
                alg_df, best_lr, metric="return", interpolation=None
            )

            xs = np.asarray(xs)
            ys = np.asarray(ys)

            assert np.all(np.isclose(xs[0], xs))

            curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            f, ax = plt.subplots()
            for i in range(xs.shape[0]):
                ax.plot(xs[i] * 10, ys[i], label=f"{alg},seed{i}", linewidth=0.5)

            ax.plot(
                np.linspace(0, exp.total_steps, 100) * 10,
                np.ones(100) * -3.86,
                "k-.",
                linewidth=1,
                label="light-on",
            )
            ax.plot(
                np.linspace(0, exp.total_steps, 100) * 10,
                np.ones(100) * -5.1,
                "b--",
                linewidth=1,
                label="random",
            )

            ax.set_xlim(0, exp.total_steps * 10)
            ax.legend()
            ax.set_title("Learning Curve in MultiPlantSimulator")
            ax.set_ylabel("Return")
            ax.set_xlabel("Day Time [Minutes]")

            save(save_path=f"{path}/plots", plot_name=f"{alg}")
            plt.show()


def auc(t, r):
    return np.sum(r)


if __name__ == "__main__":
    main()
