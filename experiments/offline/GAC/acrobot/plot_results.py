import os
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import RlEvaluation.hypers as Hypers
from PyExpPlotting.matplot import setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import TimeSummary, extract_learning_curves
from RlEvaluation.utils.pandas import split_over_column

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference("jmlr")

COLORS = {
    "GAC": "blue",
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
    # Metrics.add_step_weighted_return(df)

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col="environment"):
        f, ax = plt.subplots()
        for alg, sub_df in split_over_column(env_df, col="algorithm"):
            if len(sub_df) == 0:
                continue

            report = Hypers.select_best_hypers(
                sub_df,
                metric="return",
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
            # Plot each (x, y) pair
            plt.figure(figsize=(8, 5))
            for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
                plt.plot(x, y, marker="o", linestyle="-", label=f"Dataset {i + 1}")

            # Labels and title
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.title("All Y curves")
            plt.show()
