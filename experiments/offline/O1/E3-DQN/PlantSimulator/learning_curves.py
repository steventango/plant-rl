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

COLORS = {
    'DQN':'red'
}

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
        folder_columns=(None, None, None, None, 'environment'),
        file_col='algorithm',
    )

    assert df is not None

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        f, ax = plt.subplots(2, 1)
        for alg, sub_df in split_over_column(env_df, col='algorithm'):
            report = Hypers.select_best_hypers(
                sub_df,
                metric='action',  # picking the hypers that chose action "1" most
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            print('-' * 25)
            print(env, alg)
            Hypers.pretty_print(report)

            # Plot action history averaged over 5 seeds
            xs_a, ys_a = extract_learning_curves(sub_df, report.best_configuration, metric='action', interpolation=None)
            xs_a = np.asarray(xs_a)
            ys_a = np.asarray(ys_a)

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys_a,
                statistic=Statistic.mean,
            )

            ax[0].plot(rescale_time(xs_a[0], 1), res.sample_stat, label=alg, color=COLORS[alg], linewidth=0.5)
            ax[0].fill_between(rescale_time(xs_a[0], 1), res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
            ax[0].legend()
            ax[0].set_title('Learning curves over 12 hours')
            ax[0].set_ylabel('Action')
            ax[0].set_xlabel('Day Time [Hours]')

            # Plot reward history averaged over 5 seeds
            xs, ys = extract_learning_curves(sub_df, report.best_configuration, metric='return', interpolation=None)
            xs = np.asarray(xs)
            ys = np.asarray(ys)

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            ax[1].plot(rescale_time(xs[0],1), res.sample_stat, label=alg, color=COLORS[alg], linewidth=0.5)
            ax[1].fill_between(rescale_time(xs[0], 1), res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
            ax[1].legend()
            ax[1].set_ylabel('Accumulated Reward')
            ax[0].set_xlabel('Day Time [Hours]')


        save(save_path=f'{path}/plots', plot_name=f'{alg}', save_type='jpg')

def rescale_time(x, stride):
    base_step = 10/60           # spreadsheet time step is 10 minutes
    return x*base_step*stride   # x-values in units of hours

if __name__ == "__main__":
    main()
