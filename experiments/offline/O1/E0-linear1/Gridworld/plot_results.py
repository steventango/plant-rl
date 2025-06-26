import os  # type: ignore
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
import RlEvaluation.hypers as Hypers
from PyExpPlotting.matplot import setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import (
    TimeSummary,
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)
from RlEvaluation.utils.pandas import split_over_column

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference("jmlr")

COLORS = {
    "ESARSA": "blue",
}

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col="environment",
        algorithm_col="algorithm",
        # makes this data definition globally accessible
        # so we don't need to supply it to all API calls
        make_global=True,
    )

    df = results.combine(
        # converts path like "experiments/example/MountainCar"
        # into a new column "environment" with value "MountainCar"
        # None means to ignore a path part
        folder_columns=(None, None, None, "environment"),
        # and creates a new column named "algorithm"
        # whose value is the name of an experiment file, minus extension.
        # For instance, ESARSA.json becomes ESARSA
        file_col="algorithm",
    )

    assert df is not None

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col="environment"):
        f, ax = plt.subplots()
        for alg, sub_df in split_over_column(env_df, col="algorithm"):  # type: ignore
            if len(sub_df) == 0:
                continue

            report = Hypers.select_best_hypers(
                sub_df,  # type: ignore
                metric="return",
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            print("-" * 25)
            print(env, alg)
            Hypers.pretty_print(report)

            xs, ys = extract_learning_curves(
                sub_df,  # type: ignore
                report.best_configuration,
                metric="return",
                interpolation=None,
            )

            xs = np.asarray(xs)
            ys = np.asarray(ys)

            # make sure all of the x values are the same for each curve
            assert np.all(np.isclose(xs[0], xs))

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            ax.plot(xs[0], res.sample_stat, label=alg, color=COLORS[alg], linewidth=0.5)
            ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)

        ax.set_xlim(0, exp.total_steps)

        ax.legend()
        ax.set_title("ESARSA Gridworld")
        ax.set_ylabel("Return")
        ax.set_xlabel("Time Step")

        plt.show()
