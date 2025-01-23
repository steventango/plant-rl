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

COLORS = {
    'DQN-Relu': 'red',
}

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
    
    f, ax = plt.subplots()
    action_history = np.array(df['action'])
    reward_history = np.array(df['reward'])
    print(max(reward_history))

    which_day = 1
    ax.plot(action_history[-300:], label='action')
    ax.plot(reward_history[-300:], label='reward')
    ax.legend()
    save(
        save_path=f'{path}/plots',
        plot_name=f'DQN-Relu'
    )
    plt.show()