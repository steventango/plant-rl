from datetime import datetime, time
from pathlib import Path

import polars as pl
from minari import DataCollector

from datasets.config import GOOD_ZONE_DAYS, TIMEZONE, tzinfo
from datasets.env import MockEnv
from datasets.transforms import (
    transform_action,
    transform_action_traces,
    transform_reward,
    transform_state,
)

dfs = []
for exp_id_zone_id, good_days in GOOD_ZONE_DAYS.items():
    exp_id, zone_id = map(int, exp_id_zone_id[1:].split("/zone"))
    data_path = f"/data/online/E{exp_id}/P1/DiscreteRandom{zone_id}/alliance-zone{zone_id:02}/raw.csv"
    df = pl.read_csv(data_path, try_parse_dates=True)
    df = df.with_columns(pl.col("time").dt.convert_time_zone(TIMEZONE))
    df = df.with_columns(pl.col("time").dt.replace(second=0, microsecond=0))
    assert df.filter((pl.col("time").dt.minute() % 5 != 0)).is_empty()
    # fill in missing time steps, print how many were missing
    min_time: datetime = df["time"].min()  # type: ignore
    max_time: datetime = df["time"].max()  # type: ignore
    all_times = pl.datetime_range(min_time, max_time, interval="5m", eager=True)
    plant_ids = df.select(pl.col("plant_id")).unique()
    times_df = pl.DataFrame(data={"time": all_times})
    grid = times_df.join(plant_ids, how="cross")
    df = grid.join(df, on=["time", "plant_id"], how="left")
    df = df.sort("time", "plant_id")
    df = df.with_columns(
        pl.lit(exp_id).alias("experiment"),
        pl.lit(zone_id).alias("zone"),
    )
    df = df.filter(
        pl.col("time")
        .dt.time()
        .is_between(time(9, tzinfo=tzinfo), time(21, tzinfo=tzinfo))
    )
    df = df.sort("time", "plant_id")
    print(
        f"E{exp_id}/zone{zone_id}: missing {df['clean_area'].is_null().sum()} time steps"
    )

    df = df.with_columns(
        pl.col("action.0").fill_null(strategy="forward").over("plant_id"),
        pl.col("action.1").fill_null(strategy="forward").over("plant_id"),
        pl.col("action.2").fill_null(strategy="forward").over("plant_id"),
        pl.col("action.3").fill_null(strategy="forward").over("plant_id"),
        pl.col("action.4").fill_null(strategy="forward").over("plant_id"),
        pl.col("action.5").fill_null(strategy="forward").over("plant_id"),
        pl.col("clean_area").fill_null(strategy="forward").over("plant_id"),
    )

    df = transform_state(df)
    df = transform_action(df)
    df = transform_action_traces(df)
    df = transform_reward(df)

    df = df.with_columns(
        ((pl.col("time").dt.date() - df["time"].dt.date().min()).dt.total_days()).alias(
            "day"
        ),
    )
    df = df.with_columns(pl.col("day").is_in(good_days).alias("is_good_day"))
    print(df.select("time", "mean_clean_area", "discrete_action", "reward").describe())
    df = df.with_columns(
        (pl.col("time") == df["time"].max()).alias("terminal"),
    )
    dfs.append(df)

df = pl.concat(dfs).sort("experiment", "zone", "plant_id", "time")
print(df.head())
df_daily = df.filter(pl.col("time").dt.time() == time(9, 30, tzinfo=tzinfo))
df_daily = df_daily.filter(pl.col("is_good_day"))
df_daily = df_daily.with_columns(
    pl.col("clean_area")
    .shift(1)
    .over("experiment", "zone", "plant_id")
    .alias("prev_clean_area"),
)
df_daily = df_daily.with_columns(
    (
        (pl.col("clean_area") - pl.col("prev_clean_area")) / pl.col("prev_clean_area")
    ).alias("reward"),
)
df_daily = df_daily.drop("prev_clean_area")
# drop rows with outlier change outside 5%, 95% percentiles
percentile = 0.05
q5 = df_daily["reward"].quantile(percentile)
q95 = df_daily["reward"].quantile(1 - percentile)
df_daily_filtered = df_daily.filter(
    (pl.col("reward") >= q5) & (pl.col("reward") <= q95)
)
print(
    f"dropped {df_daily.height - df_daily_filtered.height} / {df_daily.height} daily rows as outliers"
)
print(df_daily_filtered.select("reward").describe())
print(df_daily_filtered[["time", "discrete_action", "reward"]])

# add terminal flags to daily filtered
df_daily_filtered = df_daily_filtered.with_columns(
    pl.col("time")
    .shift(-1)
    .over("experiment", "zone", "plant_id")
    .is_null()
    .alias("terminal"),
)

# add truncated flags for gaps due to outlier removal
df_daily_filtered = df_daily_filtered.with_columns(
    pl.col("time")
    .shift(-1)
    .over("experiment", "zone", "plant_id")
    .alias("next_time"),
)
df_daily_filtered = df_daily_filtered.with_columns(
    (pl.col("next_time").is_not_null() & (pl.col("next_time") != pl.col("time") + pl.duration(days=1))).alias("truncated")
)
df_daily_filtered = df_daily_filtered.drop("next_time")

# save to parquet
Path("/data/offline").mkdir(parents=True, exist_ok=True)
df.write_parquet("/data/offline/cleaned_offline_dataset.parquet")
df_daily_filtered.write_parquet("/data/offline/cleaned_offline_dataset_daily.parquet")

mock_env = MockEnv(df_daily_filtered)
env = DataCollector(mock_env, record_infos=True)

# Run episodes until environment indicates all data has been processed
while not mock_env.is_done():
    obs, info = env.reset(seed=0)

    # Check if reset indicates we're done
    if info["done"]:
        break

    while True:
        action = info["action"]
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="plant-rl/discrete-v4",
    algorithm_name="Random-Policy",
    code_permalink="https://github.com/steventango/plant-rl",
    author="Steven Tang",
    author_email="stang5@ualberta.ca",
)
