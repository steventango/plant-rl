import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import split_over_column
import RlEvaluation.hypers as Hypers
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference('neurips')

COLORS = {'tc-ESARSA': 'green'}

total_days = 7
optimal_action = np.tile(np.hstack([np.ones(3*6), 2*np.ones(6*6), np.ones(3*6)]), total_days)

def main():
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
        folder_columns=(None, None, None, 'environment'),
        file_col='algorithm',
    )

    assert df is not None

    results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        f, ax = plt.subplots(5, 1)
        for alg, sub_df in split_over_column(env_df, col='algorithm'):
            report = Hypers.select_best_hypers(
                sub_df,
                metric='return',
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            print('-' * 25)
            print(env, alg)
            Hypers.pretty_print(report)

            xs_a, ys_a = extract_learning_curves(sub_df, report.best_configuration, metric='action', interpolation=None)
            xs_a = np.asarray(xs_a)
            ys_a = np.asarray(ys_a)

            curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys_a,
                statistic=Statistic.mean,
            )

            for i in range(5):
                ax[i].plot(rescale_time(xs_a[0], 1), optimal_action, color='r', label='optimal policy', linewidth=0.5)
                ax[i].plot(rescale_time(xs_a[0], 1), ys_a[i], 'g.', label=f'seed{i+1}', markersize=0.5)
                ax[i].set_ylabel('Action')
                for j in range(total_days + 1):
                    ax[i].axvline(x = 12*j, color='k', linestyle='--', linewidth=0.5)

            ax[0].set_title(f'Learning curves over {total_days} days')
            ax[4].set_xlabel('Day Time [Hours]')


        save(save_path=f'{path}/plots', plot_name=f'{alg}', save_type='jpg')

def rescale_time(x, stride):
    base_step = 10/60           # spreadsheet time step is 10 minutes
    return x*base_step*stride   # x-values in units of hours

if __name__ == "__main__":
    main()
