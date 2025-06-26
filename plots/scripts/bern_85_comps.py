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


# Normalize mean_clean_area
for agent, group in df.groupby("agent"):
    first_5_times = (
        group[  # type: ignore
            group["time"]
            .dt.strftime("%H:%M")
            .isin(["11:40", "11:50", "12:00", "12:10", "12:20"])
        ]
        .sort_values("time")["time"]
        .unique()[:5]
    )
    # print(f"Agent: {agent}, First times: {first_5_times}")
    first_obs_values = [
        group[group["time"] == t]["mean_clean_area"].iloc[0]  # type: ignore
        for t in first_5_times  # type: ignore
    ]  # Get corresponding values
    for t in list(zip(first_5_times, first_obs_values, strict=False)):
        print(f"Agent: {agent}, Time: {t[0]}, Value: {t[1]}")
    first_obs_mean = sum(first_obs_values) / len(
        first_obs_values
    )  # Calculate mean of those values
    df.loc[group.index, "mean_clean_area"] = group["mean_clean_area"]

# Filter for unique values of time and the "Constant Bright" agent
plot_data = df[  # type: ignore
    df["agent"].isin(
        ["Bernoulli p=0.65", "Bernoulli p=0.70", "Bernoulli p=0.85", "Bernoulli p=0.90"]
    )
].drop_duplicates(subset=["time", "agent"])
plot_data["relative_time"] = plot_data.groupby("agent")["time"].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 3600
)

# Filter for the final 3 days according to day_idx
# max_day_idx = bernoulli85_data['day_idx'].max()
# final_days_data = bernoulli85_data[bernoulli85_data['day_idx'] >= max_day_idx - 2]
plot_data["new_day"] = plot_data["time"].dt.strftime("%H:%M").isin(["09:20"])
# Plot time vs mean_clean_area for the final 3 days
plt.figure(figsize=(12, 6))
# Define a color map for agents
agents = plot_data["agent"].unique()
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
# Loop through each agent and plot its data
for _i, agent in enumerate(agents):
    agent_data = plot_data[plot_data["agent"] == agent]
    plt.plot(
        agent_data["relative_time"],
        agent_data["mean_clean_area"],
        label=agent,
        color=manual_colors[agent],
    )
# Force buffer to allow text visibility
# ymin, ymax = plt.ylim()
# plt.ylim(ymin, ymax + 0.2)

plt.title("Time vs Raw Area", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Raw Area", fontsize=12)
plt.xlabel("Time Since Start (hours)", fontsize=12)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(24))  # type: ignore
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"day {int(x) // 24}")  # type: ignore
)
plt.xticks(rotation=45)

new_day_rows = plot_data[plot_data["new_day"]][
    ["time", "agent", "day_idx"]
].drop_duplicates()  # type: ignore

for _, row in new_day_rows.iterrows():
    agent = row["agent"]
    start_time = plot_data[plot_data["agent"] == agent]["time"].min()
    rel_t = (row["time"] - start_time).total_seconds() / 3600  # type: ignore
    day = int(row["day_idx"])
    plt.axvline(x=rel_t, color="red", linestyle="--")
    # plt.text(
    #     rel_t - 0.5,  # 30 minutes earlier in relative time
    #     plot_data['mean_clean_area'].max() - 0.05,
    #     f"Day {day}",
    #     color='red',
    #     fontsize=10,
    #     rotation=0,
    #     verticalalignment='bottom',
    #     horizontalalignment='right'
    # )


# noon_rows = plot_data[plot_data['time'].dt.strftime('%H:%M') == '12:00'].drop_duplicates(subset=['time', 'agent'])

# for _, row in noon_rows.iterrows():
#     agent = row['agent']
#     start_time = plot_data[plot_data['agent'] == agent]['time'].min()
#     rel_t = (row['time'] - start_time).total_seconds() / 3600
#     plt.axvline(x=rel_t, color='purple', linestyle=':', linewidth=1)
#     plt.text(
#         rel_t + 0.5,
#         plot_data['mean_clean_area'].max() - 0.05,
#         '12pm',
#         color='purple',
#         fontsize=10,
#         rotation=0,
#         verticalalignment='bottom',
#         horizontalalignment='left'
#     )


plt.legend(title="Agent", loc="upper left", fontsize=10)

plt.savefig("plots/outputs/bern_comps.png", dpi=300)
