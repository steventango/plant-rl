import os
import sys

sys.path.append(os.getcwd() + "/src")
from PyExpPlotting.matplot import setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference("neurips")

COLORS = {"tc-ESARSA": "blue"}

total_days = 14


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col="environment",
        algorithm_col="algorithm",
        make_global=True,
    )

    df = results.combine(
        folder_columns=(None, None, None, "environment"),
        file_col="algorithm",
    )

    assert df is not None

    df.to_csv(f"{path}/data.csv", index=False)

if __name__ == "__main__":
    main()
