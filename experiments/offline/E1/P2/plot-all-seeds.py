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

USE_THIS_AGENT = 'GAC-best'
COLORS = {
    USE_THIS_AGENT: 'green',
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

        for alg, sub_df in split_over_column(env_df, col='algorithm'):
            if len(sub_df) == 0: continue
            if alg == USE_THIS_AGENT:
                f, ax = plt.subplots()

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

                xs, ys = extract_learning_curves(
                        sub_df,
                        report.best_configuration,
                        metric='return',
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
                for i in range(xs.shape[0]):
                    ax.plot(xs[i], ys[i], label=f'{alg},seed{i}', linewidth=0.5)

                ax.plot(np.linspace(0, exp.total_steps, 100), np.ones(100)*202.3, 'k-.', linewidth=1, label='light-on')
                ax.plot(np.linspace(0, exp.total_steps, 100), np.ones(100)*112.5, 'b--', linewidth=1, label='random')
                ax.set_xlim(0, exp.total_steps)
                ax.legend()
                ax.set_title('Learning Curve in PlantSimulator')
                ax.set_ylabel('Return')
                ax.set_xlabel('Daytime Time Step')

                save(
                    save_path=f'{path}/plots',
                    plot_name=f'{alg}'
                )
                plt.show()
