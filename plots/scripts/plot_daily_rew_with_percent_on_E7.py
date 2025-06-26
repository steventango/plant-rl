# %%  # type: ignore
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytz

ZONE_TO_AGENT = {
    "z1": "Constant Dim",
    "z2": "Constant Bright",
    "z3": "Bandit",
    "z6": "Contextual Bandit",
    "z8": "ESARSA_alpha=0.1",
    "z9": "ESARSA_alpha=0.25",
}
datasets = []
for p in ["P2", "P3", "P4"]:
    paths = Path("/data/online/E7").joinpath(p).glob("**/z*")
    datasets.extend(sorted(paths))

dfs = []
for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    zone = dataset.name  # gets last directory name (e.g., 'z1', 'z2', etc.)
    df["agent"] = ZONE_TO_AGENT[zone]  # convert zone to agent name
    df = df[
        ["time", "agent_action", "mean_clean_area", "agent"]
    ]  # keep only needed columns
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


# Convert time column to America/Edmonton timezone
# Update the function to handle fractional seconds and timezone information
def convert_to_edmonton_timezone(utc_time):
    utc_dt = datetime.fromisoformat(
        utc_time
    )  # Use fromisoformat to parse ISO 8601 timestamps
    edmonton_tz = pytz.timezone("America/Edmonton")
    return utc_dt.astimezone(edmonton_tz)


df["time"] = df["time"].apply(convert_to_edmonton_timezone)

# Extract day from time column based on America/Edmonton timezone
df["day"] = df["time"].dt.date


# Normalize mean_clean_area by dividing by the first observed value for each agent

for _agent, group in df.groupby("agent"):
    first_obs_mean = group.sort_values("time").iloc[0]["mean_clean_area"]
    df.loc[group.index, "mean_clean_area"] = group["mean_clean_area"] / first_obs_mean

# %%
target_str = "09:05"
reward_data = []
for agent, group in df.groupby("agent"):
    group = group.sort_values("time")
    daily_groups = list(group.groupby("day"))  # Convert to list for indexing

    for i in range(len(daily_groups) - 1):  # Iterate up to the second last day
        current_day, current_group = daily_groups[i]
        next_day, next_group = daily_groups[i + 1]

        # Check if the days are consecutive
        if (next_day - current_day).days == 1:  # type: ignore
            # Filter observations at 9am for both days
            current_9am_obs = current_group[
                current_group["time"].dt.strftime("%H:%M") == target_str
            ]
            next_9am_obs = next_group[
                next_group["time"].dt.strftime("%H:%M") == target_str
            ]

            # Ensure both days have observations at 9am
            if not current_9am_obs.empty and not next_9am_obs.empty:
                current_9am_mean = current_9am_obs.iloc[0]["mean_clean_area"]
                next_9am_mean = next_9am_obs.iloc[0]["mean_clean_area"]

                diff = next_9am_mean - current_9am_mean
                reward_data.append({"agent": agent, "day": current_day, "reward": diff})

reward_df = pd.DataFrame(reward_data)

# Initialize reward column as float to avoid dtype mismatch
df["reward"] = 0.0  # Initialize reward column to 0.0


# %%
# Set reward to 0 for all entries except the last of the day for each agent
for agent, group in df.groupby("agent"):
    print(agent)
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
            }
        )

plot_df = pd.DataFrame(plot_data)

# Define agents as unique values from the plot_df dataframe
agents = plot_df["agent"].unique()

# Use the updated colormap function to avoid deprecation warning
cmap = plt.colormaps.get_cmap("tab10")
colors = [cmap(i) for i in range(len(agents))]  # Generate colors for each agent

# Filter out the last day from the plot data
plot_df = plot_df[plot_df["day"] != plot_df["day"].max()]

# Correct the x-axis labels to show each unique day without the year
plot_df["day"] = (
    plot_df["day"].astype(str).str.slice(5)  # type: ignore
)  # Extract MM-DD format from the string representation

# Group data by day and sort agents by percent_action_1 within each day
plot_data_grouped = []
for _day, group in plot_df.groupby("day"):
    sorted_group = group.sort_values(by="percent_action_1", ascending=True)  # type: ignore
    plot_data_grouped.append(sorted_group)

plot_df = pd.concat(plot_data_grouped)
# Double the group gap and bump up bar width
bar_width = 12.0  # wider bars
group_gap = 10.0  # twice as much space between day‐groups
unique_days = plot_df["day"].unique()
num_agents = len(agents)

# %%
# Keep height reasonable; just widen
fig, ax = plt.subplots(figsize=(70, 35))

for day_idx, day in enumerate(unique_days):
    base_x = day_idx * (num_agents * bar_width + group_gap)
    day_data = plot_df[plot_df["day"] == day]
    for i, (_, row) in enumerate(day_data.iterrows()):
        x = base_x + i * bar_width
        ax.bar(
            x,
            row["reward"],
            width=bar_width,
            color=colors[list(agents).index(row["agent"])],
            label=row["agent"],
        )
        ax.text(
            x,
            row["reward"] + 0.01,
            f"{row['percent_action_1']:.1f}",
            ha="center",
            rotation=45,
            fontsize=20,
        )

# Center the day‐ticks under each cluster
centers = [
    idx * (num_agents * bar_width + group_gap) + (num_agents * bar_width) / 2
    for idx in range(len(unique_days))
]
ax.set_xticks(centers)
ax.set_xticklabels(unique_days, rotation=45)

ax.set_xlabel("Days", fontsize=40)
# ax.set_ylabel('Pct change in \n area', rotation=0, labelpad=140, fontsize=20)
ax.set_ylabel(
    "Daily change in area \n (normalized by inital plant size)",
    rotation=0,
    labelpad=300,
    fontsize=40,
)

ax.set_title("Daily Change in Area by Agent with Proportion(Action == 1)", fontsize=60)

# Clean up duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles, strict=False))
ax.legend(
    by_label.values(), by_label.keys(), title="Agent", fontsize=60, title_fontsize=20
)

plt.tight_layout()
plt.savefig("plots/outputs/barplots_E7.png", dpi=300)

# %%


# Create a grid of line plots with dots for datapoints
unique_days = plot_df["day"].unique()
num_days = len(unique_days)

# Set up the grid dimensions
cols = 4
rows = (num_days + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), sharex=False, sharey=False)
axes = axes.flatten()

legend_handles = {}
i = -1

if num_days > 0:
    for i, day in enumerate(unique_days):
        ax = axes[i]
        day_data = plot_df[plot_df["day"] == day]

        # Black line for all dots of the day
        ax.plot(
            day_data["percent_action_1"],
            day_data["reward"],
            linestyle="-",
            linewidth=1,
            color="black",
            alpha=1.0,
        )

        # Dots per agent
        for agent in agents:
            agent_data = day_data[day_data["agent"] == agent]
            scatter = ax.scatter(
                agent_data["percent_action_1"],
                agent_data["reward"],
                color=colors[list(agents).index(agent)],
                s=70,
                edgecolor="none",
                label=agent,
            )
            legend_handles[agent] = scatter

        ax.set_title(f"Day {day}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Shared axis labels
fig.text(0.5, 0.04, "Proportion Action == 1", ha="center", fontsize=14)
fig.text(
    0.04,
    0.5,
    "Daily change in area (normalized by initial plant size)",
    va="center",
    rotation="vertical",
    fontsize=14,
)

# Shared legend
fig.legend(
    legend_handles.values(),
    legend_handles.keys(),
    title="Agent",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(agents),
    fontsize=12,
    title_fontsize=14,
)

plt.tight_layout(
    rect=[0.04, 0.04, 1, 0.95]
)  # Adjust for space for legend and labels  # type: ignore
plt.savefig("plots/outputs/grid_line_plots_E7.png", dpi=300)

# %%
