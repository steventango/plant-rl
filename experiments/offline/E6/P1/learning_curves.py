import os
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.interpolation import compute_step_return
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import (
    TimeSummary,
    curve_percentile_bootstrap_ci,
    extract_multiple_learning_curves,
)
from RlEvaluation.utils.pandas import split_over_column

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference("jmlr")

COLORS = {
    "SoftmaxAC": "grey",
    "EQRC": "blue",
    "ESARSA": "red",
    "DQN": "black",
    "PrioritizedDQN": "purple",
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

    f, ax = plt.subplots()
    for lambda_val, lambda_df in split_over_column(df, col="lambda"):
        alphas = []
        best_scores = []
        for alpha_val, sub_df in sorted(split_over_column(lambda_df, col="alpha"), key=lambda x: x[0]):
            sub_df = sub_df[sub_df["episode"] < 50]
            print(sub_df["return"].count())
            score = -np.nanmean(sub_df["return"])
            print(lambda_val, alpha_val, f"{score:.2f}")
            best_scores.append(score)
            alphas.append(alpha_val)

        ax.plot(alphas, best_scores, label=lambda_val)
    ax.set_xlim(0, 1.75)
    ax.set_ylim(170, 400)
    ax.set_title(
        "Mountain Car\n"
        r"ESARSA($\lambda$) with replacing traces"
    )
    ax.set_xlabel(r"$\alpha \times$ number of tilings (8)")
    ax.set_ylabel("Mountain Car\nSteps per episode\n averaged over\nfirst 50 episodes\nand 30 runs", rotation=0)
    ax.yaxis.set_label_coords(-0.2, 0.5)

    ax.legend()
    save(save_path=f"{path}/plots", plot_name=f"ESARSA(lambda) with replacing traces")
    plt.clf()
