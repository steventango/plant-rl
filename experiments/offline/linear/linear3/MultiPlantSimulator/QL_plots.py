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
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference('jmlr')

COLORS = {
    'QL': 'red',
    'constant':'black',
    'ESARSA': 'blue',
}
def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=['alpha', 'epsilon', 'decay_eps_frac'],
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

    for env, env_df in split_over_column(df, col='environment'):
        for lag, lag_df in split_over_column(env_df, col='environment.lag'):
            for stride, stride_df in split_over_column(lag_df, col='environment.stride'):
                f_all, ax_all = plt.subplots()
                for alg, alg_df in split_over_column(stride_df, col='algorithm'):
                    hyper2metric = {}
                    for alpha in alg_df['alpha'].unique():
                        for epsilon in alg_df['epsilon'].unique():
                            for decay_eps_frac in alg_df['decay_eps_frac'].unique():
                                xs, ys = extract_learning_curves(alg_df, (alpha, epsilon, decay_eps_frac), metric='return', interpolation=None)
                                metric = [r[-1] for r in ys]
                                hyper2metric[(alpha, epsilon, decay_eps_frac)] = np.mean(metric)

                    best_hyper = max(hyper2metric, key=hyper2metric.get)
                    print(f'Best hypers for {alg} with stride {stride} and lag {lag}: {best_hyper}')

                    # Reward Plot
                    xs, ys = extract_learning_curves(alg_df, best_hyper, metric='return', interpolation=None)
                    xs = np.asarray(xs)
                    ys = np.asarray(ys)

                    res_r = curve_percentile_bootstrap_ci(
                        rng=np.random.default_rng(0),
                        y=ys,
                        statistic=Statistic.mean,
                    )

                    f, ax = plt.subplots(figsize=(10, 8))

                    # Action Plot
                    xs_a, ys_a = extract_learning_curves(alg_df, best_hyper, metric='action', interpolation=None)
                    xs_a = np.asarray(xs_a)
                    ys_a = np.asarray(ys_a)

                    res_a = curve_percentile_bootstrap_ci(
                        rng=np.random.default_rng(0),
                        y=ys_a,
                        statistic=Statistic.mean,
                    )

                    ax.plot(rescale_time(xs_a[0], stride), res_a.sample_stat, color=COLORS[alg], linewidth=0.8, label='Mean Action')
                    ax.fill_between(rescale_time(xs_a[0], stride), res_a.ci[0], res_a.ci[1], color=COLORS[alg], alpha=0.2)
                    ax.set_xlim(0, 168)
                    ax.set_ylabel('Action')
                    ax.set_xlabel('Day Time [Hours]')
                    ax.legend()

                    save(
                        save_path=f'{path}/plots',
                        plot_name=f'{alg}_actions_stride={stride}_lag={lag}_{best_hyper}'
                    )

                    plt.close(f)

                    ax_all.plot(rescale_time(xs[0], stride), res_r.sample_stat, label=f'{alg}', color=COLORS[alg], linewidth=0.5)
                    ax_all.fill_between(rescale_time(xs[0], stride), res_r.ci[0], res_r.ci[1], color=COLORS[alg], alpha=0.2)

                ax_all.set_xlim(0, 168)
                ax_all.set_title(f'All Algorithms (stride={stride} lag={lag})')
                ax_all.set_ylabel('Accumulated Reward')
                ax_all.legend()

                save(
                    save_path=f'{path}/plots',
                    plot_name=f'all_algos_stride={stride}_lag={lag}'
                )

                plt.close(f_all)

    


def rescale_time(x, stride):
    base_step = 10/60           # spreadsheet time step is 10 minutes
    return x*base_step*stride   # x-values in units of hours

def rescale_return(y, min, max):
    return (y - min) / (max - min)

if __name__ == "__main__":
    main()