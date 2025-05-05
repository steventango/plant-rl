import os
import sys
sys.path.append(os.getcwd() + '/src')
os.environ['NUMBA_DISABLE_JIT'] = '1'

import enum
import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
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

setDefaultConference('neurips')

total_days = 12

THIS_AGENT = 'Context_Bandit'

W0 = 0.1
EP = 0.1

def last_n_percent_sum(x, n=0.2):
    return np.nansum(x[:, int((1-n)*x.shape[1]):], axis=1)

class TimeSummary(enum.Enum):
    last_n_percent_sum = enum.member(last_n_percent_sum)

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
           
    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        for alg, alg_df in split_over_column(env_df, col='algorithm'):   
            if alg == THIS_AGENT:
                sub_df = alg_df
                sub_df = alg_df[(alg_df['w0']==W0) & (alg_df['epsilon']==EP) &  (alg_df['alpha']==0.1)]
                report = Hypers.select_best_hypers(
                    sub_df,
                    metric='reward', 
                    prefer=Hypers.Preference.high,
                    time_summary=TimeSummary.last_n_percent_sum,
                    statistic=Statistic.mean,
                )

                print('-' * 25)
                print(env, alg)
                Hypers.pretty_print(report)

                #xs, ys = extract_learning_curves(sub_df, report.best_configuration, metric='return', interpolation=None)
                #f_r, ax_r = plt.subplots(5, 1)
                #for i in range(5):
                #    ax_r[i].plot(xs[i], ys[i], '.')
                #plt.show()
                
                xs_w0, ys_w0 = extract_learning_curves(sub_df, report.best_configuration, metric='weight0', interpolation=None)
                xs_w1, ys_w1 = extract_learning_curves(sub_df, report.best_configuration, metric='weight1', interpolation=None)
                xs_w2, ys_w2 = extract_learning_curves(sub_df, report.best_configuration, metric='weight2', interpolation=None)
                xs_w3, ys_w3 = extract_learning_curves(sub_df, report.best_configuration, metric='weight3', interpolation=None)
                
                steps_per_day = int(len(xs_w0[0])/total_days)
                f, ax = plt.subplots(5, 1, figsize=(5, 6))

                # Find global vmin and vmax across all data
                all_data = []
                for i in range(5):
                    ys_w = np.vstack([ys_w3[i][3*steps_per_day:], ys_w2[i][3*steps_per_day:], ys_w1[i][3*steps_per_day:], ys_w0[i][3*steps_per_day:]])
                    all_data.append(ys_w)

                # Compute global min and max
                global_vmin = min(np.min(data) for data in all_data)
                global_vmax = max(np.max(data) for data in all_data)
                print(global_vmin, global_vmax)

                # Plot with consistent vmin and vmax
                for i in range(5):
                    ys_w = np.vstack([ys_w3[i], ys_w2[i], ys_w1[i], ys_w0[i]])
                    
                    im = ax[i].imshow(ys_w, aspect=40, extent=(0, ys_w.shape[1], 0, 3), 
                                    vmin=global_vmin, vmax=global_vmax, cmap='Purples')
                    ax[i].set_ylabel('Action')       
                    ax[i].set_xlim([0, ys_w.shape[1]])
                    for j in range(total_days + 1):
                        ax[i].axvline(x = steps_per_day*j, color='k', linestyle='--', linewidth=0.5)

                cbar_ax = f.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
                cbar = f.colorbar(im, cax=cbar_ax, orientation='horizontal')
                ax[0].set_title(f"{alg} Value function; w0={W0}, ep={EP}; best_score={report.best_score:.3f}")
                ax[4].set_xlabel('Daytime Steps')
                save(save_path=f'{path}/plots', plot_name=f'{alg}-w0={W0}-ep={EP}', save_type='jpg')

def rescale_time(x, stride):
    base_step = 10/60           # spreadsheet time step is 10 minutes
    return x*base_step*stride   # x-values in units of hours

if __name__ == "__main__":
    main()