# %%  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator, HourLocator


def normalize(x, lower, upper):
    return (x - lower) / (upper - lower)


df_1: pd.DataFrame = pd.read_csv("/data/online/E7/P2/Constant1/z1/raw.csv")
df_1 = df_1[df_1["plant_id"] == 0]
df_2: pd.DataFrame = pd.read_csv("/data/online/E7/P2/Constant2/z2/raw.csv")
df_2 = df_2[df_2["plant_id"] == 0]

# Convert time columns to datetime
df_1["time"] = pd.to_datetime(df_1["time"])
df_2["time"] = pd.to_datetime(df_2["time"])

# Convert time to America/Edmonton timezone
df_1["time"] = df_1["time"].dt.tz_convert("America/Edmonton")  # type: ignore
df_2["time"] = df_2["time"].dt.tz_convert("America/Edmonton")  # type: ignore

# %%
df_1["mean_clean_area"]
# %% compute alternative rewards
reward_cols = [
    "reward",
    "reward_20min",
    "reward_30min",
    "reward_1hour",
    "reward_2hour",
    "reward_4hour",
    "reward_6hour",
    "reward_1day",
]
for df in [df_1, df_2]:
    # Ensure mean_clean_area is a pandas Series  # type: ignore
    mean_clean_area: pd.Series = df["mean_clean_area"]  # type: ignore
    df["reward_20min"] = mean_clean_area - mean_clean_area.shift(2)  # type: ignore
    df["reward_30min"] = mean_clean_area - mean_clean_area.shift(3)  # type: ignore
    df["reward_1hour"] = mean_clean_area - mean_clean_area.shift(6)  # type: ignore
    df["reward_2hour"] = mean_clean_area - mean_clean_area.shift(12)  # type: ignore
    df["reward_4hour"] = mean_clean_area - mean_clean_area.shift(24)  # type: ignore
    df["reward_6hour"] = mean_clean_area - mean_clean_area.shift(36)
    df["reward_1day"] = mean_clean_area - mean_clean_area.shift(72)
    for reward_col in reward_cols:  # type: ignore
        if reward_col != "reward":
            df[reward_col] = df[reward_col].apply(lambda x: normalize(x, 0, 50))
# %%
df_1.head()

# %%

# Add a source column to identify each dataset
df_1["source"] = "Constant Dim"
df_2["source"] = "Constant Standard"

# Combine the dataframes for easier plotting
combined_df = pd.concat([df_1, df_2], ignore_index=True)

# Set up the plot with Seaborn styling
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
# type: ignore
# Create the timeseries plot with datetime x-axis  # type: ignore
sns.lineplot(data=df_1, x="time", y="mean_clean_area", label="Constant Dim")
sns.lineplot(data=df_2, x="time", y="mean_clean_area", label="Constant Standard")

# Format the x-axis to show dates nicely
ax = plt.gca()

# Set up locators for ticks at day boundaries and at 9am/9pm
days = DayLocator(tz="America/Edmonton")
hours = HourLocator(byhour=[9, 21], tz="America/Edmonton")  # 9am and 9pm

# Format the date labels
date_format = DateFormatter("%Y-%m-%d", tz="America/Edmonton")
time_format = DateFormatter("%H:%M", tz="America/Edmonton")

# Apply the locators
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_minor_formatter(time_format)

# Rotate labels for better readability
plt.xticks(rotation=90, ha="right")
plt.xticks(rotation=90, ha="right", minor=True)

# Add title and labels
plt.title("Mean Clean Area Over Time", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Mean Clean Area", fontsize=12)
plt.legend(title="Data Source")

# Improve the appearance
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# %%
# Create a shifted version of df_2 with time shifted forward by 1 day
df_2_shifted = df_2.copy()
df_2_shifted["time"] = df_2_shifted["time"] + pd.Timedelta(days=1)  # type: ignore
# Ensure timezone is preserved in the shifted dataframe  # type: ignore
if df_2_shifted["time"].dt.tz is None:  # type: ignore
    df_2_shifted["time"] = df_2_shifted["time"].dt.tz_localize("America/Edmonton")  # type: ignore
elif str(df_2_shifted["time"].dt.tz) != "America/Edmonton":
    df_2_shifted["time"] = df_2_shifted["time"].dt.tz_convert("America/Edmonton")

df_2_shifted["source"] = "Constant Standard (shifted +1 day)"

# Set up the time-shifted comparison plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot original df_1 and shifted df_2  # type: ignore
sns.lineplot(
    data=df_1, x="time", y="mean_clean_area", label="Constant Dim", legend=False
)  # type: ignore
sns.lineplot(
    data=df_2_shifted,
    x="time",
    y="mean_clean_area",
    label="Constant Standard (shifted +1 day)",
    legend=False,
)

# Format the x-axis to show dates nicely
ax = plt.gca()
days = DayLocator(tz="America/Edmonton")
hours = HourLocator(byhour=[9, 21], tz="America/Edmonton")
date_format = DateFormatter("%Y-%m-%d", tz="America/Edmonton")
time_format = DateFormatter("%H:%M", tz="America/Edmonton")
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_minor_formatter(time_format)
plt.xticks(rotation=90, ha="right")
plt.xticks(rotation=90, ha="right", minor=True)

# Add title and labels
plt.title("Mean Clean Area Over Time (Constant Standard Shifted +1 Day)", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Mean Clean Area", fontsize=12)
plt.annotate(
    "Constant Dim",
    xy=(0, 0.1),
    xycoords="axes fraction",
    fontsize=12,
    color=ax.get_lines()[0].get_color(),
)
plt.annotate(
    "Constant Standard\n(shifted +1 day)",
    xy=(0.8, 0.9),
    xycoords="axes fraction",
    fontsize=12,
    color=ax.get_lines()[1].get_color(),
)

# remove splines
sns.despine()

# Improve the appearance
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# %%
# Plot reward timeseries for the shifted timeseries
# Set up the reward time-shifted comparison plot as subfigures
fig, axes = plt.subplots(
    nrows=len(reward_cols), ncols=1, figsize=(12, 6 * len(reward_cols)), sharex=True
)
sns.set_style("whitegrid")

for i, reward_col in enumerate(reward_cols):
    ax = axes[i]

    # Plot rewards for the current reward column  # type: ignore
    sns.lineplot(
        data=df_1,
        x="time",
        y=reward_col,
        label=f"Constant Dim - {reward_col}",
        alpha=0.8,
        ax=ax,
    )  # type: ignore
    sns.lineplot(
        data=df_2_shifted,
        x="time",
        y=reward_col,
        label=f"Constant Standard (shifted +1 day) - {reward_col}",
        alpha=0.8,
        ax=ax,
    )

    # Format the x-axis to show dates nicely
    days = DayLocator(tz="America/Edmonton")
    hours = HourLocator(byhour=[9, 21], tz="America/Edmonton")
    date_format = DateFormatter("%Y-%m-%d", tz="America/Edmonton")
    time_format = DateFormatter("%H:%M", tz="America/Edmonton")
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_minor_locator(hours)
    ax.xaxis.set_minor_formatter(time_format)
    ax.tick_params(axis="x", rotation=90)

    # Add title and labels
    ax.set_title(
        f"{reward_col} Over Time (Constant Standard Shifted +1 Day)", fontsize=14
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(reward_col, fontsize=12)

    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3)

# Improve the appearance
plt.tight_layout()
plt.show()


# %%

df_2_shifted["episode"] = df_2_shifted["episode"] + 1
combined_df_shifted = pd.concat([df_1, df_2_shifted], ignore_index=True)

# Alternative visualization using violin plots for better distribution visualization
plt.figure(figsize=(14, 8))  # type: ignore

sns.violinplot(data=combined_df_shifted, x="episode", y="reward", hue="source")
plt.title("Reward Distribution by Episode - Constant Dim", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%

df_analysis = df_2_shifted.copy()
df_analysis = df_analysis.reset_index(drop=True)


# %%
df_1_skip = df_1.copy()  # type: ignore
df_1_skip = df_1_skip[
    df_1_skip["time"] - df_analysis["time"].min() > -pd.Timedelta(minutes=5)  # type: ignore
]
df_1_skip = df_1_skip.reset_index(drop=True)
# %%
for reward_col in reward_cols:
    df_analysis[f"{reward_col}_delta"] = df_analysis[reward_col] - df_1_skip[reward_col]
# omit the last two episode
df_analysis = df_analysis[df_analysis["episode"] < 8]

# %%  # type: ignore
# explode the dataframe to have one row per reward delta, and a column to indicate the type of reward delta
df_analysis = df_analysis.melt(
    id_vars=["episode", "time", "mean_clean_area", "source"],
    value_vars=[f"{reward_col}_delta" for reward_col in reward_cols],
    var_name="reward_type",
    value_name="reward_delta_value",
)
# %%

# %%
# plot violin plots of reward delta for each episode  # type: ignore
n_episodes = df_analysis["episode"].nunique()  # type: ignore
fig, axes = plt.subplots(
    nrows=n_episodes, ncols=1, figsize=(6, 4 * n_episodes), sharex=True
)

for i, episode in enumerate(sorted(df_analysis["episode"].unique())):
    episode_df = df_analysis[df_analysis["episode"] == episode]
    ax = axes[i]  # type: ignore
    sns.violinplot(
        data=episode_df,
        x="reward_type",
        y="reward_delta_value",
        hue="reward_type",
        ax=ax,
    )
    ax.set_title(f"Episode {episode}", fontsize=12)
    ax.set_xlabel("Reward Type", fontsize=10)
    if i == 0:
        ax.set_ylabel("Reward Delta", fontsize=10)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
# %%
