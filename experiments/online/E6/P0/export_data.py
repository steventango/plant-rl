import os
import sys

sys.path.append(os.getcwd() + "/src")
import numpy as np
import pandas as pd
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.utils.pandas import split_over_column

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs


def maybe_convert_to_array(x):
    if isinstance(x, float) or isinstance(x, int):
        return x
    if isinstance(x, bytes):
        return np.frombuffer(x)
    return x


def main():
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    dd = data_definition(
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

    for metric in ["area", "state", "action", "reward"]:
        df[metric] = df[metric].apply(maybe_convert_to_array)

    df.to_csv(f"{path}/data.csv", index=False)


if __name__ == "__main__":
    main()
