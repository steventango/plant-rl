#!/usr/bin/env python3
"""
Plot reward as a function of discrete action.

Usage:
  python plot.py --parquet /data/offline/cleaned_offline_dataset_daily.parquet \
                 --action-col action --reward-col reward --out reward_by_action.png
"""
import argparse
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_column(df: pl.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot reward vs discrete action from a parquet dataset.")
    parser.add_argument("--parquet", "-p", default="/data/offline/cleaned_offline_dataset_daily.parquet",
                        help="Path to parquet file (default: %(default)s)")
    parser.add_argument("--action-col", "-a", default="discrete_action", help="Column name for discrete action")
    parser.add_argument("--reward-col", "-r", default="reward", help="Column name for reward")
    parser.add_argument("--out", "-o", default="reward_by_action.png", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    try:
        logging.info("Reading parquet: %s", args.parquet)
        df = pl.read_parquet(args.parquet)
    except Exception as e:
        logging.error("Failed to read parquet: %s", e)
        sys.exit(1)

    # detect common column names if not provided
    action_candidates = ["action", "actions", "discrete_action", "action_id", "action_idx", "act"]
    reward_candidates = ["reward", "rewards", "return", "score"]
    action_col = args.action_col or find_column(df, action_candidates)
    reward_col = args.reward_col or find_column(df, reward_candidates)

    if action_col is None or reward_col is None:
        logging.error("Could not find action or reward columns automatically.")
        logging.info("Available columns: %s", df.columns)
        logging.info("Provide --action-col and --reward-col to override.")
        sys.exit(1)

    logging.info("Using action column: %s, reward column: %s", action_col, reward_col)

    # sort by time to ensure correct shifting
    df = df.sort("time", "experiment", "zone", "plant_id")
    df = df.with_columns(pl.col(reward_col).shift(-1).over("experiment", "zone", "plant_id").alias(reward_col))
    # convert to pandas for seaborn
    pdf = df[[action_col, reward_col]].to_pandas()

    # Ensure action is treated as a discrete/categorical variable
    # Convert category values to strings so palette keys match reliably
    if pd.api.types.is_integer_dtype(pdf[action_col]) or pd.api.types.is_numeric_dtype(pdf[action_col]):
        # If floats but represent discrete integers, convert safely
        if pd.api.types.is_float_dtype(pdf[action_col]):
            # check if values are integers
            if (pdf[action_col].dropna() % 1 == 0).all():
                pdf[action_col] = pdf[action_col].astype("Int64")
        pdf[action_col] = pdf[action_col].astype("category")
    else:
        pdf[action_col] = pdf[action_col].astype("category")

    # normalize category labels to strings so we can map colors reliably
    pdf[action_col] = pdf[action_col].astype(str).astype("category")

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # define palette: 0 -> red, 1 -> white, 2 -> blue
    palette = {"0": "red", "1": "white", "2": "blue", "nan": "gray"}

    # boxplot to show distribution per action (use custom palette)
    ax = sns.violinplot(x=action_col, y=reward_col, data=pdf, palette=palette)

    # overlay mean reward per action as a pointplot with 95% CI
    sns.pointplot(
        x=action_col,
        y=reward_col,
        data=pdf,
        estimator="mean",
        errorbar=('ci', 95),
        color="black",
        capsize=0.1,
        errwidth=1
    )

    plt.title("Reward by Discrete Action")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    logging.info("Saved plot to %s", args.out)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
