import os
import sys

sys.path.append(os.getcwd() + "/src")

import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

from RlEvaluation.config import data_definition
from RlEvaluation.temporal import (
    TimeSummary,
    extract_learning_curves,
    curve_percentile_bootstrap_ci,
)
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import split_over_column

import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference("jmlr")

COLORS = {
    "DQN-1Relu": "red",
    "DQN-2Relu": "green",
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
    Metrics.add_step_weighted_return(df)

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col="environment"):
        f, ax = plt.subplots()
        for alg, sub_df in split_over_column(env_df, col="algorithm"):
            if len(sub_df) == 0:
                continue

            report = Hypers.select_best_hypers(
                sub_df,
                metric="step_weighted_return",
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
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

            # make sure all of the x values are the same for each curve
            assert np.all(np.isclose(xs[0], xs))

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            ax.plot(xs[0], res.sample_stat, label=alg, color=COLORS[alg], linewidth=0.5)
            ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)

        ax.plot(
            np.linspace(0, exp.total_steps, 100),
            np.ones(100) * 13.75,
            "k--",
            linewidth=1,
            label="return of light-on policy",
        )
        ax.plot(
            np.linspace(0, exp.total_steps, 100),
            np.ones(100) * 3.75,
            "b--",
            linewidth=1,
            label="return of random policy",
        )
        ax.set_xlim(0, exp.total_steps)
        ax.set_ylim(0, 14.5)
        ax.legend()
        ax.set_title("Compare Different Representation Networks in PlantSimulator")
        ax.set_ylabel("Return")
        ax.set_xlabel("Daytime Time Step")

        save(save_path=f"{path}/plots", plot_name="rep-nets")
        plt.show()
