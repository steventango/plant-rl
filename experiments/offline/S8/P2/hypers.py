import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import json

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from PyExpUtils.results.tools import getParamsAsDict
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


def main():
    results = ResultCollection(Model=ExperimentModel, metrics=["return"])
    results.paths = [path for path in results.paths if "best" not in path]
    print("results paths", results.paths)
    data_definition(
        hyper_cols=["id"],
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    for env, sub_results in results.groupby_directory(level=4):
        for alg_result in sub_results:
            alg = alg_result.filename
            print(alg)

            df = alg_result.load()
            if df is None:
                continue
            df = df.with_columns((pl.col("id") // 720).alias("seed"))
            df = df.with_columns((pl.col("id") % 720).alias("cid"))
            df = df.sort(["cid", "seed"])

            df = df.with_columns(pl.col("returns").arr.mean().alias("mean_return"))
            print(df)
            df2 = df.filter(pl.col("steps") >= 90000)
            # calculate mean_return over id and seed
            df2 = df2.group_by(["cid", "seed"]).agg(pl.col("mean_return").mean())
            # calculate mean_Return over cid
            df3 = df2.group_by("cid").agg(pl.col("mean_return").mean())
            # sort by mean_return
            df3 = df3.sort("mean_return", descending=True)
            print(alg)
            print(df3)

            if len(df2) == 0:
                continue

            best_config_id = df3["cid"][0]

            params = getParamsAsDict(alg_result.exp, best_config_id)
            with open(alg_result.exp_path.replace(".json", "_best.json"), "w") as f:
                json.dump(params, f, indent=4)

            best_df = df2.filter(pl.col("cid") == best_config_id)
            best_df = best_df.select("cid", "seed", "mean_return")
            # add id col back
            best_df = best_df.with_columns(
                (pl.col("cid") + pl.col("seed") * 720).alias("id")
            )
            print(best_df)

            # Plot the mean return of the best config
            best_df = df.filter(pl.col("cid") == best_config_id)
            pdf = best_df.to_pandas()
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=pdf, x="steps", y="mean_return")
            plt.title(f"Mean Return of Best Config ({best_config_id})")
            plt.xlabel("Steps")
            plt.ylabel("Mean Return")
            plt.tight_layout()
            plt.savefig(f"{alg}_best_return.png")
            plt.close()
            print(f"Saved plot to {alg}_best_return.png")


if __name__ == "__main__":
    main()
