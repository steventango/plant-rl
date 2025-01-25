import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

from RlEvaluation.config import data_definition
from RlEvaluation.interpolation import compute_step_return
from RlEvaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import split_over_column

import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',
        make_global=True,
    )

    df = results.combine(
        folder_columns=(None, None, 'environment'),
        file_col='algorithm',
    )

    grouped_df = df.groupby(["optimizer.alpha", "seed"])[["return", "frame"]]

    plt.figure(figsize=(8, 5))
    COLORS = ['r', 'g', 'b']
    for (alpha, seed), sub_df in grouped_df:
        sub_df = sub_df.dropna(subset=["return"])
        
        if alpha == 0.001:
            plt.plot(sub_df["frame"], sub_df["return"], color=COLORS[0], linewidth=0.5)
        elif alpha == 0.0001:
            plt.plot(sub_df["frame"], sub_df["return"], color=COLORS[1], linewidth=0.5)
        elif alpha == 0.00001:
            plt.plot(sub_df["frame"], sub_df["return"], color=COLORS[2], linewidth=0.5)
        
    plt.xlabel("Time Step")
    plt.ylabel("Return")
    plt.legend()
    plt.show()