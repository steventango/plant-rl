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

PLOT_THIS = 'GAC-sweep-s12'

COLORS = {
    'GAC-sweep-s2': 'red',
    'GAC-sweep-s3': 'orange',
    'GAC-sweep-s6': 'green',
    'GAC-sweep-s9': 'blue',
    'GAC-sweep-s12': 'purple',
}

MAX_RETURN =  {
    'GAC-sweep-s2': 2.702,
    'GAC-sweep-s3': 2.679,
    'GAC-sweep-s6': 2.622,
    'GAC-sweep-s9': 2.567,
    'GAC-sweep-s12': 2.518,
}

MIN_RETURN =  {
    'GAC-sweep-s2': 0.053,
    'GAC-sweep-s3': 0.047,
    'GAC-sweep-s6': 0.044,
    'GAC-sweep-s9': 0.051,
    'GAC-sweep-s12': 0.056,
}

STRIDE = {
    'GAC-sweep-s2': 2,
    'GAC-sweep-s3': 3,
    'GAC-sweep-s6': 6,
    'GAC-sweep-s9': 9,
    'GAC-sweep-s12': 12,
}

def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=['critic_lr', 'actor_lr_scale'],
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

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        for alg, alg_df in split_over_column(env_df, col='algorithm'):            
            if alg == PLOT_THIS:
                f, ax = plt.subplots(2, 1)
                
                # Pick the best hypers by total reward over 14 days (1 episode)
                hyper2metric = {}
                for clr in alg_df['critic_lr'].unique():
                    for alr in alg_df['actor_lr_scale'].unique():
                        xs, ys = extract_learning_curves(alg_df, (clr, alr), metric='return', interpolation=None)
                        print(len(xs))
                        assert len(xs) == 5  
                        metric = [r[-1] for r in ys]
                        hyper2metric[(clr, alr)] = np.mean(metric)
                
                best_hyper = max(hyper2metric, key=hyper2metric.get) 
                print(f'Best hypers for {alg} = {best_hyper}')
                
                # Plot reward history averaged over 5 seeds
                xs, ys = extract_learning_curves(alg_df, best_hyper, metric='return', interpolation=None)
                xs = np.asarray(xs)
                ys = np.asarray(ys)

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                )
                
                ax[0].plot(rescale_time(xs[0], STRIDE[alg]), res.sample_stat, label=f'time step={10*STRIDE[alg]}min', color=COLORS[alg], linewidth=0.5)
                ax[0].fill_between(rescale_time(xs[0], STRIDE[alg]), res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
                ax[0].set_xlim(0, 168)
                ax[0].set_title('Learning Curves')
                ax[0].set_ylabel('Accumulated Reward')

                # Plot action history averaged over 5 seeds
                xs_a, ys_a = extract_learning_curves(alg_df, best_hyper, metric='action', interpolation=None)
                xs_a = np.asarray(xs_a)
                ys_a = np.asarray(ys_a)
                
                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys_a,
                    statistic=Statistic.mean,
                )

                ax[1].plot(rescale_time(xs_a[0], STRIDE[alg]), res.sample_stat, label=f'time step={10*STRIDE[alg]}min', color=COLORS[alg], linewidth=0.5)
                ax[1].fill_between(rescale_time(xs_a[0], STRIDE[alg]), res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
                ax[1].set_xlim(0, 168)
                ax[1].legend()
                ax[1].set_ylabel('Action')
                ax[1].set_xlabel('Day Time [Hours]')

                save(
                    save_path=f'{path}/plots',
                    plot_name=f'{alg}'
                )

def rescale_time(x, stride):
    base_step = 10/60           # spreadsheet time step is 10 minutes
    return x*base_step*stride   # x-values in units of hours

def rescale_return(y, min, max):
    return (y - min) / (max - min)

if __name__ == "__main__":
    main()