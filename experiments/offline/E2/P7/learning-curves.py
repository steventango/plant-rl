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
    "GAC-sweep-s2": "red",
    "GAC-sweep-s3": "orange",
    "GAC-sweep-s6": "green",
    "GAC-sweep-s9": "blue",
    "GAC-sweep-s12": "purple",
}

MAX_RETURN = {
    "GAC-sweep-s2": 2.702,
    "GAC-sweep-s3": 2.679,
    "GAC-sweep-s6": 2.622,
    "GAC-sweep-s9": 2.567,
    "GAC-sweep-s12": 2.518,
}

MIN_RETURN = {
    "GAC-sweep-s2": 0.053,
    "GAC-sweep-s3": 0.047,
    "GAC-sweep-s6": 0.044,
    "GAC-sweep-s9": 0.051,
    "GAC-sweep-s12": 0.056,
}

STRIDE = {
    "GAC-sweep-s2": 2,
    "GAC-sweep-s3": 3,
    "GAC-sweep-s6": 6,
    "GAC-sweep-s9": 9,
    "GAC-sweep-s12": 12,
}


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

    results.get_any_exp()

    for _env, env_df in split_over_column(df, col="environment"):
        f, ax = plt.subplots()
        for alg, alg_df in split_over_column(env_df, col="algorithm"):  # type: ignore
            # Pick the best learning rate by AUC
            lr2metric = {}
            for lr in alg_df["critic_lr"].unique():  # type: ignore
                xs, ys = extract_learning_curves(
                    alg_df,
                    (lr,),
                    metric="return",
                    interpolation=None,  # type: ignore
                )
                assert len(xs) == 5
                metric = [auc(t, r) for t, r in zip(xs, ys, strict=False)]
                lr2metric[lr] = np.mean(metric)

            best_lr = max(lr2metric, key=lr2metric.get)  # type: ignore
            print(f"Best critic_lr = {best_lr}")
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
                rescale_time(xs[0], STRIDE[alg]),
                rescale_return(res.sample_stat, MIN_RETURN[alg], MAX_RETURN[alg]),
                label=f"time step={10 * STRIDE[alg]}min",
                color=COLORS[alg],
                linewidth=0.5,
            )
            ax.fill_between(
                rescale_time(xs[0], STRIDE[alg]),
                rescale_return(res.ci[0], MIN_RETURN[alg], MAX_RETURN[alg]),
                rescale_return(res.ci[1], MIN_RETURN[alg], MAX_RETURN[alg]),
                color=COLORS[alg],
                alpha=0.2,
            )

        ax.plot(
            np.linspace(0, 5000, 100),
            np.ones(100),
            "k--",
            linewidth=0.5,
            label="light-on",
        )
        ax.set_ylim(0.35, 1.02)
        ax.set_xlim(0, 5000)
        ax.legend()
        ax.set_title("GAC's Learning Curves in PlantSimulator (32 plants)")
        ax.set_ylabel(
            "Normalized Episodic Return [light-off policy: 0, light-on policy: 1]"
        )
        ax.set_xlabel("Day Time [Hours]")

        save(save_path=f"{path}/plots", plot_name="GAC-2action-32plant")


def rescale_time(x, stride):
    base_step = 10 / 60  # spreadsheet time step is 10 minutes
    return x * base_step * stride  # x-values in units of hours


def rescale_return(y, min, max):
    return (y - min) / (max - min)


def auc(t, r):
    return np.sum(r)


if __name__ == "__main__":
    main()
