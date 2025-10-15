from itertools import product

import polars as pl

from datasets.config import BLUE, RED, WHITE


def transform_action(df: pl.DataFrame) -> pl.DataFrame:
    # shift action backwards
    df = df.with_columns(pl.col("action.0").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.1").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.2").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.3").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.4").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.5").shift(-1).over("plant_id"))

    groups = df.group_by("time", "plant_id")
    groups_with_counts = groups.agg(pl.len())
    # Assert that the count for all groups is 1
    assert (groups_with_counts["len"] == 1).all()

    df2 = groups.agg(
        pl.col("action.0").mean(),
        pl.col("action.1").mean(),
        pl.col("action.2").mean(),
        pl.col("action.3").mean(),
        pl.col("action.4").mean(),
        pl.col("action.5").mean(),
        pl.col("clean_area").mean(),
    ).sort("time", "plant_id")
    df2 = df2.with_columns(
        pl.concat_arr(
            [
                pl.col("action.0"),
                pl.col("action.1"),
                pl.col("action.2"),
                pl.col("action.3"),
                pl.col("action.4"),
                pl.col("action.5"),
            ]
        ).alias("action")
    )
    df2 = df2.with_columns(
        red_diff=(pl.col("action") - RED[None])
        .arr.to_list()
        .list.eval(pl.element().abs())
        .list.sum(),
        white_diff=(pl.col("action") - WHITE[None])
        .arr.to_list()
        .list.eval(pl.element().abs())
        .list.sum(),
        blue_diff=(pl.col("action") - BLUE[None])
        .arr.to_list()
        .list.eval(pl.element().abs())
        .list.sum(),
    )
    eps = 0.1
    df2 = df2.with_columns(
        pl.when(pl.col("red_diff") < eps)
        .then(0)
        .when(pl.col("white_diff") < eps)
        .then(1)
        .when(pl.col("blue_diff") < eps)
        .then(2)
        .otherwise(None)
        .alias("discrete_action")
    )
    df = df.join(
        df2.select(["time", "plant_id", "discrete_action"]),
        on=["time", "plant_id"],
        how="left",
    )
    return df


def transform_reward(df):
    df = df.with_columns(
        pl.col("clean_area").shift(1).over("plant_id").alias("prev_clean_area"),
    )
    df = df.with_columns(
        (
            (pl.col("clean_area") - pl.col("clean_area")) / pl.col("prev_clean_area")
        ).alias("reward"),
    )
    df = df.drop("prev_clean_area")
    return df


def transform_action_traces(df):
    df = df.sort("plant_id", "time")
    action_cols = [
        "action.0",
        "action.1",
        "action.2",
        "action.3",
        "action.4",
        "action.5",
    ]
    alphas = [0.3, 0.5, 0.7]
    for col, alpha in product(action_cols, alphas):
        df = df.with_columns(
            pl.col(col)
            .ewm_mean(alpha=alpha, adjust=True)
            .over("plant_id")
            .alias(f"{col}_trace_{alpha}"),
        )
    # Create one-hot for discrete_action
    df = df.with_columns(
        pl.when(pl.col("discrete_action") == 0)
        .then(1.0)
        .otherwise(0.0)
        .alias("discrete_action_0"),
        pl.when(pl.col("discrete_action") == 1)
        .then(1.0)
        .otherwise(0.0)
        .alias("discrete_action_1"),
        pl.when(pl.col("discrete_action") == 2)
        .then(1.0)
        .otherwise(0.0)
        .alias("discrete_action_2"),
    )
    for alpha in alphas:
        df = df.with_columns(
            pl.col("discrete_action_0")
            .ewm_mean(alpha=alpha, adjust=True)
            .over("plant_id")
            .alias(f"discrete_action_trace_0_{alpha}"),
            pl.col("discrete_action_1")
            .ewm_mean(alpha=alpha, adjust=True)
            .over("plant_id")
            .alias(f"discrete_action_trace_1_{alpha}"),
            pl.col("discrete_action_2")
            .ewm_mean(alpha=alpha, adjust=True)
            .over("plant_id")
            .alias(f"discrete_action_trace_2_{alpha}"),
        )
    return df


def transform_state(df):
    df = df.with_columns(
        pl.col("clean_area").mean().over("plant_id").alias("mean_clean_area"),
    )
    return df
