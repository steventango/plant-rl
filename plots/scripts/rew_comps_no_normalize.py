# %%
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
import pytz

matplotlib.use(
    "Agg"
)  # Prevent X server requirement (useful when running headless or via SSH)
import matplotlib.pyplot as plt

ZONE_TO_AGENT_E7 = {
    "z1": "Constant Dim",
    "z2": "Constant Bright",
    "z3": "Bandit",
    "z6": "Contextual Bandit",
    "z8": "ESARSA_alpha=0.1",
    "z9": "ESARSA_alpha=0.25",
}

ZONE_TO_AGENT_E8 = {
    "z2": "Bernoulli p=0.90",
    "z3": "Bernoulli p=0.85",
    "z8": "Bernoulli p=0.70",
    "z9": "Bernoulli p=0.65",
}


dfs = []

datasets = []
for p in ["P1"]:
    paths = Path("/data/online/E8").joinpath(p).glob("Bernoulli*/z*")
    datasets.extend(sorted(paths))
for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    zone = dataset.name  # gets last directory name (e.g., 'z1', 'z2', etc.)
    df["agent"] = ZONE_TO_AGENT_E8[zone]  # convert zone to agent name
    df = df[
        ["time", "agent_action", "mean_clean_area", "agent"]
    ]  # keep only needed columns
    dfs.append(df)

datasets = []
for p in ["P2", "P3", "P4"]:
    paths = Path("/data/online/E7").joinpath(p).glob("**/z*")
    datasets.extend(sorted(paths))

for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    zone = dataset.name  # gets last directory name (e.g., 'z1', 'z2', etc.)
    df["agent"] = ZONE_TO_AGENT_E7[zone]  # convert zone to agent name
    df = df[
        ["time", "agent_action", "mean_clean_area", "agent"]
    ]  # keep only needed columns
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


# Convert time column to America/Edmonton timezone
# Update the function to handle fractional seconds and timezone information
def convert_to_edmonton_timezone(utc_time):
    if isinstance(utc_time, str):
        utc_dt = datetime.fromisoformat(utc_time)
    else:
        utc_dt = utc_time
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    edmonton_tz = pytz.timezone("America/Edmonton")
    return utc_dt.astimezone(edmonton_tz)


df["time"] = df["time"].apply(convert_to_edmonton_timezone)

# Extract day from time column based on America/Edmonton timezone
df["day"] = df["time"].dt.date

# Create day_idx column for each agent group
for agent, group in df.groupby("agent"):
    # Sort by time to ensure correct day ordering
    sorted_group = group.sort_values("time")
    # Print unique times for each agent
    # print(f"\nAgent: {agent}")
    # print(sorted_group['time'].dt.strftime('%H:%M').unique())
    # Get unique days and create a mapping to indices
    first_day = min(sorted_group["day"])
    # Calculate day_idx as the number of days since the first day
    df.loc[sorted_group.index, "day_idx"] = (sorted_group["day"] - first_day).apply(
        lambda x: x.days
    )


# # Normalize mean_clean_area
# for agent, group in df.groupby('agent'):
#     first_5_times = group[group['time'].dt.strftime('%H:%M').isin(['11:40', '11:50', '12:00', '12:10', '12:20'])].sort_values('time')['time'].unique()[:5]
#     #print(f"Agent: {agent}, First times: {first_5_times}")
#     first_obs_values = [group[group['time'] == t]['mean_clean_area'].iloc[0] for t in first_5_times]  # Get corresponding values
#     for t in list(zip(first_5_times, first_obs_values)):
#         print(f"Agent: {agent}, Time: {t[0]}, Value: {t[1]}")
#     first_obs_mean = sum(first_obs_values) / len(first_obs_values)  # Calculate mean of those values
#     df.loc[group.index, 'mean_clean_area'] = group['mean_clean_area'] / first_obs_mean

# %%
reward_data = []
target_times = [
    "11:40",
    "11:50",
    "12:00",
    "12:10",
    "12:20",
]  # Target times for 12pm observations
for agent, group in df.groupby("agent"):
    group = group.sort_values("time")
    daily_groups = list(group.groupby("day"))  # Convert to list for indexing

    for i in range(len(daily_groups) - 1):  # Iterate up to the second last day
        current_day, current_group = daily_groups[i]
        next_day, next_group = daily_groups[i + 1]
        print(
            f"Processing agent: {agent}, current day: {current_day}, next day: {next_day}"
        )
        # Check if the days are consecutive
        if (next_day - current_day).days == 1:
            print(f"Found consecutive days: {current_day} and {next_day}")

            # Max 5 obs
            current_max_obs = current_group.drop_duplicates(subset=["time"]).nlargest(
                5, "mean_clean_area"
            )
            next_max_obs = next_group.drop_duplicates(subset=["time"]).nlargest(
                5, "mean_clean_area"
            )
            print(
                f"Current day max observations at times {current_max_obs['time'].dt.strftime('%H:%M').values}:"
            )
            print(f"Values: {current_max_obs['mean_clean_area'].values}")
            print(
                f"Next day max observations at times {next_max_obs['time'].dt.strftime('%H:%M').values}:"
            )
            print(f"Values: {next_max_obs['mean_clean_area'].values}")
            has_max_obs = not current_max_obs.empty and not next_max_obs.empty

            # Average of 5 obs centered around 12pm
            current_12pm_obs = current_group[
                current_group["time"].dt.strftime("%H:%M").isin(target_times)
            ].drop_duplicates(subset=["time"])
            next_12pm_obs = next_group[
                next_group["time"].dt.strftime("%H:%M").isin(target_times)
            ].drop_duplicates(subset=["time"])
            print(
                f"Number of observations in current_12pm_obs: {len(current_12pm_obs)}"
            )
            print(f"Number of observations in next_12pm_obs: {len(next_12pm_obs)}")
            has_12pm_obs = not current_12pm_obs.empty and not next_12pm_obs.empty

            # 9am obs
            current_9am_obs = current_group[
                current_group["time"].dt.strftime("%H:%M").isin(["09:20"])
            ].drop_duplicates(subset=["time"])
            next_9am_obs = next_group[
                next_group["time"].dt.strftime("%H:%M").isin(["09:20"])
            ].drop_duplicates(subset=["time"])
            print(f"Number of observations in current_9am_obs: {len(current_9am_obs)}")
            print(f"Number of observations in next_9am_obs: {len(next_9am_obs)}")
            has_9am_obs = not current_9am_obs.empty and not next_9am_obs.empty

            # Ensure both days have observations
            if has_max_obs and has_12pm_obs and has_9am_obs:
                current_max = current_max_obs["mean_clean_area"].mean()
                next_max = next_max_obs["mean_clean_area"].mean()
                reward_max = next_max - current_max

                current_12pm_mean = current_12pm_obs["mean_clean_area"].mean()
                next_12pm_mean = next_12pm_obs["mean_clean_area"].mean()
                reward_12_avg = next_12pm_mean - current_12pm_mean

                current_9am = current_9am_obs.iloc[0]["mean_clean_area"]
                next_9am = next_9am_obs.iloc[0]["mean_clean_area"]
                reward_9 = next_9am - current_9am

                reward_data.append(
                    {
                        "agent": agent,
                        "day": current_day,
                        "reward_max": reward_max,
                        "reward_12_avg": reward_12_avg,
                        "reward_9": reward_9,
                    }
                )
            else:
                print(
                    f"Skipping agent {agent} for days {current_day} and {next_day} due to missing observations."
                )
                continue

reward_df = pd.DataFrame(reward_data)
# Initialize reward column as float to avoid dtype mismatch

df["reward_max"] = 0.0  # Initialize reward column to 0.0
df["reward_12_avg"] = 0.0
df["reward_9"] = 0.0

# %%
# Set reward to 0 for all entries except the last of the day for each agent
for agent, group in df.groupby("agent"):
    # print(agent)
    daily_groups = group.groupby("day")
    for day, daily_group in daily_groups:
        last_obs_index = daily_group.index[
            -1
        ]  # Get index of the last observation of the day
        reward_row = reward_df.loc[
            (reward_df["agent"] == agent) & (reward_df["day"] == day)
        ]
        if not reward_row.empty:  # Check if reward_row is not empty
            # Set the reward values for the last observation of the day
            for col in ["reward_max", "reward_12_avg", "reward_9"]:
                df.loc[last_obs_index, col] = reward_row[col].values[0]

# %%
# Group data by day and calculate reward and percentage of agent_action == 1
plot_data = []
for (day, agent), group in df.groupby(["day", "agent"]):
    reward_max = group[
        "reward_max"
    ].max()  # Get the reward for the last observation of the day
    reward_12_avg = group["reward_12_avg"].max()
    reward_9 = group["reward_9"].max()
    percent_action_1 = (
        group["agent_action"] == 1
    ).mean()  # Calculate percentage of agent_action == 1
    if reward_max != 0 and reward_12_avg != 0 and reward_9 != 0:
        plot_data.append(
            {
                "day": day,
                "agent": agent,
                "percent_action_1": percent_action_1,
                "day_idx": group["day_idx"].iloc[0],
                "reward_max": reward_max,
                "reward_12_avg": reward_12_avg,
                "reward_9": reward_9,
            }
        )

plot_df = pd.DataFrame(plot_data)

# %%
# Calculate global y-axis limits for the first figure
global_ymin = plot_df[["reward_max", "reward_12_avg", "reward_9"]].min().min()
global_ymax = plot_df[["reward_max", "reward_12_avg", "reward_9"]].max().max()

# Create a figure with three subplots for reward_max, reward_12_avg, and reward_9
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# Define the reward columns and titles for the subplots
reward_columns = ["reward_max", "reward_12_avg", "reward_9"]
titles = [
    "Rew = ∆ avg of max 5 areas",
    "Rew = ∆ avg of 5 areas centered around 12pm",
    "Rew = ∆ area at 9:20am",
]
agents = plot_df["agent"].unique()  # Get the unique agents

# Loop through each reward column and create a subplot
for ax, reward_col, title in zip(axes, reward_columns, titles, strict=False):
    for agent in agents:
        agent_data = plot_df[plot_df["agent"] == agent]
        ax.plot(agent_data["day_idx"], agent_data[reward_col], label=agent, marker="o")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("Reward")
    ax.legend(title="Agent", fontsize=8, title_fontsize=10)
    ax.set_ylim(global_ymin, global_ymax)  # Set the same y-axis limits for all subplots

# Add a title above all subplots
fig.suptitle("Reward Comparison (raw)", fontsize=16, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
plt.savefig("plots/outputs/reward_comparison_by_type_no_norm.png", dpi=300)

# Calculate global y-axis limits for the second figure
global_ymin = plot_df[["reward_max", "reward_12_avg", "reward_9"]].min().min()
global_ymax = plot_df[["reward_max", "reward_12_avg", "reward_9"]].max().max()

# Define the reward columns and titles for the subplots
agents = plot_df["agent"].unique()  # Get the unique agents
n_agents = len(agents)
reward_columns = ["reward_max", "reward_12_avg", "reward_9"]

cols = 4
rows = (n_agents + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes = axes.flatten()

# Loop through each reward column and create a subplot
for ax, agent in zip(axes, agents, strict=False):
    agent_data = plot_df[plot_df["agent"] == agent]
    for reward in reward_columns:
        ax.plot(agent_data["day_idx"], agent_data[reward], label=reward, marker="o")
    ax.set_title(agent, fontsize=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("Reward")
    ax.legend(title="Reward", fontsize=8, title_fontsize=10)
    ax.set_ylim(global_ymin, global_ymax)  # Set the same y-axis limits for all subplots

for ax in axes:
    ax.tick_params(axis="both", which="both", labelsize=10)  # Ensure labels are visible

# Remove unused axes
for j in range(len(agents), len(axes)):
    fig.delaxes(axes[j])

# Add a title above all subplots
fig.suptitle("Reward Comparison by Agent (raw)", fontsize=16, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
plt.savefig("plots/outputs/reward_comparison_by_agent_no_norm.png", dpi=300)
