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

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

COLORS = {
    'Random': 'red',
    'GAC': 'green',
    'GACP': 'blue',
    'Constant': 'black',
}

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)
    hyper_cols = results.get_hyperparameter_columns()
    hyper_cols.remove('action')
    data_definition(
        hyper_cols=hyper_cols,
        seed_col='seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',

        # makes this data definition globally accessible
        # so we don't need to supply it to all API calls
        make_global=True,
    )

    df = results.combine(
        # converts path like "experiments/example/MountainCar"
        # into a new column "environment" with value "MountainCar"
        # None means to ignore a path part
        folder_columns=(None, None, None, 'environment'),

        # and creates a new column named "algorithm"
        # whose value is the name of an experiment file, minus extension.
        # For instance, ESARSA.json becomes ESARSA
        file_col='algorithm',
    )

    assert df is not None

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        f, ax = plt.subplots()
        for alg, sub_df in split_over_column(env_df, col='algorithm'):
            print(alg)
            if len(sub_df) == 0: continue

            report = Hypers.select_best_hypers(
                sub_df,
                metric='reward',
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            print('-' * 25)
            print(env, alg)
            Hypers.pretty_print(report)

            xs, ys = extract_learning_curves(
                    sub_df,
                    report.best_configuration,
                    metric='reward',
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

            ax.plot(xs[0], res.sample_stat, label=alg, color=COLORS[alg], linewidth=1)

            for x, y in zip(xs, ys, strict=False):
                ax.plot(x, y, color=COLORS[alg], linewidth=0.5, alpha=0.2)

        ax.set_xlim(0, exp.total_steps)
        # Set minor ticks every 1
        minor_ticks = np.arange(0, exp.total_steps + 1, 1)
        ax.set_xticks(minor_ticks, minor=True)
        # Set major ticks and labels every 5
        major_ticks = np.arange(0, exp.total_steps + 1, 5)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_ticks)

        # Style minor ticks (optional)
        ax.tick_params(axis='x', which='minor', length=4, color='gray')
        ax.legend()
        ax.set_title('PlantGrowthChamber')
        ax.set_ylabel('Reward')
        ax.set_xlabel('Step (minute)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        save(
            save_path=f'{path}/plots',
            plot_name='algs'
        )
