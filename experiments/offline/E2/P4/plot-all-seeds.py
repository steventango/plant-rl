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

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference("jmlr")

THIS_AGENT = "GAC-sweep"
COLORS = {THIS_AGENT: "green"}


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=["critic_lr"],
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
        f, ax = plt.subplots()
        for alg, alg_df in split_over_column(env_df, col="algorithm"):  # type: ignore
            # Pick the best learning rate
            lr2metric = {}
            for lr in alg_df["critic_lr"].unique():  # type: ignore
                xs, ys = extract_learning_curves(
                    alg_df,
                    (lr,),
                    metric="return",
                    interpolation=None,  # type: ignore
                )
                assert len(xs) == 5  # check all 5 seeds are there
                metric = [auc(t, r) for t, r in zip(xs, ys, strict=False)]
                lr2metric[lr] = np.mean(metric)

            best_lr = max(lr2metric, key=lr2metric.get)  # type: ignore
            xs, ys = extract_learning_curves(
                alg_df,
                (best_lr,),
                metric="return",
                interpolation=None,  # type: ignore
            )

            xs = np.asarray(xs)
            ys = np.asarray(ys)

            assert np.all(np.isclose(xs[0], xs))

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            ax.plot(
                xs[0], res.sample_stat, label=f"{alg}", color=COLORS[alg], linewidth=1
            )

            for i in range(xs.shape[0]):
                ax.plot(xs[i], ys[i], color=COLORS[alg], linewidth=0.5, alpha=0.2)

            ax.plot(
                np.linspace(0, exp.total_steps, 100),
                np.ones(100) * 14.059,
                "k-.",
                linewidth=1,
                label="light-on",
            )
            ax.plot(
                np.linspace(0, exp.total_steps, 100),
                np.ones(100) * 3.629,
                "b--",
                linewidth=1,
                label="random",
            )

            ax.legend()
            ax.set_title("Learning Curve in MultiPlantSimulator")
            ax.set_ylabel("Return")
            ax.set_xlabel("Day Time [Hours]")

            save(save_path=f"{path}/plots", plot_name=f"{alg}")
            plt.show()


def auc(t, r):
    return np.sum(r)


if __name__ == "__main__":
    main()
