# %%  # type: ignore
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
for _agent, group in df.groupby("agent"):
    # Sort by time to ensure correct day ordering
    sorted_group = group.sort_values("time")
    # Print unique times for each agent
    # print(f"\nAgent: {_agent}")
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

target_times = ["11:40", "11:50", "12:00", "12:10", "12:20"]  # Target times
reward_data = []
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
        if (next_day - current_day).days == 1:  # type: ignore
            print(f"Found consecutive days: {current_day} and {next_day}")
            # Filter observations for target times and drop duplicates
            current_12pm_obs = current_group[  # type: ignore
                current_group["time"].dt.strftime("%H:%M").isin(target_times)
            ].drop_duplicates(subset=["time"])
            next_12pm_obs = next_group[  # type: ignore
                next_group["time"].dt.strftime("%H:%M").isin(target_times)
            ].drop_duplicates(subset=["time"])
            print(
                f"Number of observations in current_12pm_obs: {len(current_12pm_obs)}"
            )
            print(f"Number of observations in next_12pm_obs: {len(next_12pm_obs)}")

            # Ensure both days have observations at 12pm
            if not current_12pm_obs.empty and not next_12pm_obs.empty:
                current_12pm_mean = current_12pm_obs["mean_clean_area"].mean()
                next_12pm_mean = next_12pm_obs["mean_clean_area"].mean()

                diff = next_12pm_mean - current_12pm_mean
                reward_data.append({"agent": agent, "day": current_day, "reward": diff})

reward_df = pd.DataFrame(reward_data)
# Initialize reward column as float to avoid dtype mismatch
df["reward"] = 0.0  # Initialize reward column to 0.0


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
            df.loc[last_obs_index, "reward"] = reward_row["reward"].values[0]

# %%
# Group data by day and calculate reward and percentage of agent_action == 1
plot_data = []
for (day, agent), group in df.groupby(["day", "agent"]):  # type: ignore
    reward = group["reward"].max()  # Get the reward for the last observation of the day
    percent_action_1 = (
        group["agent_action"] == 1
    ).mean()  # Calculate percentage of agent_action == 1
    if reward > 0:
        plot_data.append(
            {
                "day": day,
                "agent": agent,
                "reward": reward,
                "percent_action_1": percent_action_1,
                "day_idx": group["day_idx"].iloc[0],
            }
        )

plot_df = pd.DataFrame(plot_data)

# Define agents as unique values from the plot_df dataframe
agents = plot_df["agent"].unique()

# Manually define unique colors for agents, using names from ZONE_TO_AGENT_E7 and ZONE_TO_AGENT_E8
manual_colors = {
    "Constant Dim": "blue",
    "Constant Bright": "orange",
    "Bandit": "green",
    "Contextual Bandit": "purple",
    "ESARSA_alpha=0.1": "red",
    "ESARSA_alpha=0.25": "brown",
    "Bernoulli p=0.90": "pink",
    "Bernoulli p=0.85": "cyan",
    "Bernoulli p=0.70": "gold",
    "Bernoulli p=0.65": "lime",
}

# Filter out the last day from the plot data
plot_df = plot_df[plot_df["day"] != plot_df["day"].max()]

# Correct the x-axis labels to show each unique day without the year
plot_df["day"] = (
    plot_df["day"].astype(str).str.slice(5)  # type: ignore
)  # Extract MM-DD format from the string representation

# Group data by day and sort agents by percent_action_1 within each day
plot_data_grouped = []
for _day, group in plot_df.groupby("day_idx"):
    sorted_group = group.sort_values(by="percent_action_1", ascending=True)  # type: ignore
    plot_data_grouped.append(sorted_group)

plot_df = pd.concat(plot_data_grouped)

# %%
# Create a grid of line plots with dots for datapoints
unique_days = plot_df["day_idx"].unique()
num_days = len(unique_days)
num_agents = len(agents)

global_ymin = plot_df["reward"].min()
global_ymax = plot_df["reward"].max() * 1.2


# Set up the grid dimensions
cols = 4  # Number of columns in the grid
rows = (num_days + cols - 1) // cols  # Calculate rows based on number of days
fig, axes = plt.subplots(rows, cols, figsize=(22, 5 * rows), sharex=False, sharey=False)
axes = axes.flatten()

# Initialize `i` to handle cases where `unique_days` is empty
i = -1

# Create a line plot for each day
if num_days > 0:
    for i, day in enumerate(unique_days):
        ax = axes[i]
        day_data = plot_df[plot_df["day_idx"] == day]

        ax.set_xlim(0.5, 1)
        ax.set_ylim(
            global_ymin, global_ymax
        )  # Set the same y-axis limits for all subplots

        # Add a black line connecting all dots for the day
        ax.plot(
            day_data["percent_action_1"],
            day_data["reward"],
            linestyle="-",  # Solid line
            linewidth=1,  # Thicker line for connecting all dots
            color="black",  # Neutral color for the connecting line
            alpha=1.0,
        )  # Full opacity for visibility

        # Update dot colors based on agent type
        for agent in agents:
            agent_data = day_data[day_data["agent"] == agent]
            color = manual_colors.get(
                agent, "gray"
            )  # Use manual color mapping, default to 'gray' for unknown agents
            ax.scatter(
                agent_data["percent_action_1"],
                agent_data["reward"],
                color=color,  # Use specific colors for each agent
                s=70,  # Size of the dots
                edgecolor="none",  # Solid dots without edge
                label=agent,
            )  # Add label for legend

        ax.set_title(f"Day {int(day)}", fontsize=12)
        ax.set_xlabel("Proportion Action == Bright")
        ax.set_ylabel("Daily raw change in area \n")
        # ax.legend(title="Agent", fontsize=8, title_fontsize=10)

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


handles, labels = axes[i].get_legend_handles_labels()
by_label = dict(zip(labels, handles, strict=False))

# Add global legend to top right (outside)
fig.legend(
    by_label.values(),
    by_label.keys(),
    title="Agent",
    title_fontsize=14,
    fontsize=12,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
)

plt.tight_layout(rect=(0, 0, 0.85, 0.97))
fig.suptitle(
    "Percent Bright vs 24 hr change in raw area (avg of 5 obs centered on 12pm)",
    fontsize=18,
)
plt.savefig("plots/outputs/grid_line_plots_bern_only_12_avg.png", dpi=300)

# %%
