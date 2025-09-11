# %%  # type: ignore
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
import pytz
import numpy as np

matplotlib.use(
    "Agg"
)  # Prevent X server requirement (useful when running headless or via SSH)
import matplotlib.pyplot as plt

# ZONE_TO_AGENT_E7 = {
#     "z1": "Constant Dim",
#     "z2": "Constant Bright",
#     "z3": "Bandit",
#     "z6": "Contextual Bandit",
#     "z8": "ESARSA_alpha=0.1",
#     "z9": "ESARSA_alpha=0.25",
# }

# ZONE_TO_AGENT_E8 = {
#     "z2": "Bernoulli p=0.90",
#     "z3": "Bernoulli p=0.85",
#     "z8": "Bernoulli p=0.70",
#     "z9": "Bernoulli p=0.65",
# }
ACTION_TO_COLOR = {
    0: 'White',
    1: 'Blue',
    2: 'Red',
}

# Convert agent_action from string array format to first element
def parse_agent_action(x):
    if isinstance(x, str) and x.startswith('['):
        # Remove brackets and split by whitespace, take first element
        return 'Night'
    elif isinstance(x, str):
        return ACTION_TO_COLOR[int(x)]
    elif isinstance(x, int):
        return ACTION_TO_COLOR[x]
    else:
        print(type(x))
        raise ValueError("Invalid agent_action format")

dfs = []

datasets = []
for p in ["P1"]:
    paths = Path("/data/online/E11").joinpath(p).glob("DiscreteRandom*/alliance-z*")
    datasets.extend(sorted(paths))
for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    zone = dataset.name  # gets last directory name (e.g., 'alliance-zone01', 'alliance-zone02', etc.)
    zone_num = int(zone.split('-')[-1].replace('zone', ''))  # extract the number after removing 'zone' prefix and convert to int
    df['zone'] = zone_num
    # df["agent"] = ZONE_TO_AGENT_E8[zone]  # convert zone to agent name
    
    df['agent_action'] = df['agent_action'].apply(parse_agent_action)
    df = df[
        ["time", "agent_action", "mean_clean_area", 'zone']
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

# Create day_idx column for each zone group
# for zone, group in df.groupby("zone"):
#     # Sort by time to ensure correct day ordering
#     sorted_group = group.sort_values("time")
#     # Print unique times for each agent
#     # print(f"\nAgent: {_agent}")
#     # print(sorted_group['time'].dt.strftime('%H:%M').unique())
#     # Get unique days and create a mapping to indices
#     first_day = min(sorted_group["day"])
#     # Calculate day_idx as the number of days since the first day
#     df.loc[sorted_group.index, "day_idx"] = (sorted_group["day"] - first_day).apply(
#         lambda x: x.days
#     )


# Normalize mean_clean_area
for zone, group in df.groupby('zone'):
    first_time = group[group['time'].dt.strftime('%H:%M').isin(['09:30'])].sort_values('time')['time'].unique()
    #print(f"Zone: {zone}, First times: {first_time}")
    first_obs_values = [group[group['time'] == t]['mean_clean_area'].iloc[0] for t in first_time]  # Get corresponding values
    for t in list(zip(first_time, first_obs_values)):
        print(f"Zone: {zone}, Time: {t[0]}, Value: {t[1]}")
    first_obs_mean = sum(first_obs_values) / len(first_obs_values)  # Calculate mean of those values
    df.loc[group.index, 'mean_clean_area'] = group['mean_clean_area'] / first_obs_mean



# target_times = ['11:40', '11:50', '12:00', '12:10', '12:20']  # Target times
target_times = [
    "09:30",
]
reward_data = []
for zone, group in df.groupby("zone"):
    group = group.sort_values("time")
    daily_groups = list(group.groupby("day"))  # Convert to list for indexing

    for i in range(len(daily_groups) - 1):  # Iterate up to the second last day
        current_day, current_group = daily_groups[i]
        next_day, next_group = daily_groups[i + 1]
        print(
            f"Processing zone: {zone}, current day: {current_day}, next day: {next_day}"
        )
        # Check if the days are consecutive
        if (next_day - current_day).days == 1:  # type: ignore
            print(f"Found consecutive days: {current_day} and {next_day}")
            # Filter observations for target times and drop duplicates
            current_9am_obs = current_group[  # type: ignore
                current_group["time"].dt.strftime("%H:%M").isin(target_times)
            ].drop_duplicates(subset=["time"])
            next_9am_obs = next_group[  # type: ignore
                next_group["time"].dt.strftime("%H:%M").isin(target_times)
            ].drop_duplicates(subset=["time"])
            print(f"Number of observations in current_9am_obs: {len(current_9am_obs)}")
            print(f"Number of observations in next_9am_obs: {len(next_9am_obs)}")

            # Ensure both days have observations at 12pm
            if not current_9am_obs.empty and not next_9am_obs.empty:
                current_9am_mean = current_9am_obs["mean_clean_area"].mean()
                next_9am_mean = next_9am_obs["mean_clean_area"].mean()

                # Get agent_action from 10:00am for current group
                current_10am_obs = current_group[  # type: ignore
                    current_group["time"].dt.strftime("%H:%M") == "10:00"
                ]
                action = current_10am_obs["agent_action"].iloc[0] if not current_10am_obs.empty else None

                diff = (next_9am_mean - current_9am_mean) / current_9am_mean
                reward_data.append({"zone": zone, "day": current_day, "reward": diff, "action": action})

reward_df = pd.DataFrame(reward_data)

# Filter out observations from day 20
reward_df_filtered = reward_df[reward_df['day'].apply(lambda x: x.day not in [20, 29, 3, 4, 5])]

# Print all entries of each column in reward_df_filtered
print("=" * 60)
print("REWARD_DF_FILTERED CONTENTS:")
print("=" * 60)
print(f"Total rows: {len(reward_df_filtered)}")
print()

for column in reward_df_filtered.columns:
    print(f"Column: {column}")
    print("-" * 30)
    for idx, value in reward_df_filtered[column].items():
        print(f"  Row {idx}: {value}")
    print()

print("=" * 60)

# Calculate average reward for each action
action_rewards = reward_df_filtered.groupby('action')['reward'].mean()
action_counts = reward_df_filtered.groupby('action').size()
action_std = reward_df_filtered.groupby('action')['reward'].std()

# Calculate 95% bootstrapped confidence intervals
import scipy.stats as stats
from scipy.stats import bootstrap

confidence_intervals = []
for action in action_rewards.index:
    action_data = reward_df_filtered[reward_df_filtered['action'] == action]['reward'].values
    if len(action_data) > 1:  # Need more than 1 observation for confidence interval
        # Bootstrap confidence interval
        def mean_statistic(x):
            return np.mean(x)
        
        res = bootstrap((action_data,), mean_statistic, n_resamples=5000, 
                       confidence_level=0.95, random_state=42)
        ci_lower, ci_upper = res.confidence_interval
        mean_val = np.mean(action_data)
        # Calculate symmetric error bar size (average of distances to upper and lower bounds)
        ci = (ci_upper - ci_lower) / 2
    else:
        ci = 0
    confidence_intervals.append(ci)

# Create bar plot for average reward by action
plt.figure(figsize=(10, 6))
actions = list(action_rewards.index)
avg_rewards = list(action_rewards.values)

# Use colors from ACTION_TO_COLOR mapping
action_colors = {'White': 'lightgray', 'Blue': 'blue', 'Red': 'red', 'Night': 'black'}
colors = [action_colors.get(action, 'gray') for action in actions]

bars = plt.bar(actions, avg_rewards, color=colors, yerr=confidence_intervals, 
               capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

# Add sample size labels on top of each bar (accounting for error bars)
for i, (action, bar) in enumerate(zip(actions, bars)):
    sample_size = action_counts[action]
    bar_height = bar.get_height()
    ci = confidence_intervals[i]
    plt.text(bar.get_x() + bar.get_width()/2, bar_height + ci + 0.005, 
             f'n={sample_size}', ha='center', va='bottom', fontsize=10)

plt.title('Average Daily Change in Area by Action (Normalized)\nwith 95% Bootstrapped Confidence Intervals', fontsize=14)
plt.xlabel('Action', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.xticks(fontsize=10)
plt.savefig("plots/average_reward_by_action.png", dpi=300, bbox_inches="tight")

# Create plot split by day (one subplot per day)
unique_days = sorted(reward_df_filtered['day'].unique())
n_days = len(unique_days)

# Calculate subplot layout (try to make it roughly square)
n_cols = int(np.ceil(np.sqrt(n_days)))
n_rows = int(np.ceil(n_days / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
if n_rows == 1 and n_cols == 1:
    axes = [axes]
elif n_rows == 1 or n_cols == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

action_colors = {'White': 'lightgray', 'Blue': 'blue', 'Red': 'red', 'Night': 'black'}

for i, day in enumerate(unique_days):
    if i < len(axes):
        ax = axes[i]
        
        # Filter data for this specific day
        day_data = reward_df_filtered[reward_df_filtered['day'] == day]
        
        if not day_data.empty:
            # Calculate statistics for this day
            day_action_rewards = day_data.groupby('action')['reward'].mean()
            day_action_counts = day_data.groupby('action').size()
            day_action_std = day_data.groupby('action')['reward'].std()
            
            # Calculate bootstrapped confidence intervals for this day
            day_confidence_intervals = []
            for action in day_action_rewards.index:
                action_data = day_data[day_data['action'] == action]['reward'].values
                if len(action_data) > 1:  # Need more than 1 observation for confidence interval
                    # Bootstrap confidence interval
                    def mean_statistic(x):
                        return np.mean(x)
                    
                    res = bootstrap((action_data,), mean_statistic, n_resamples=5000, 
                                   confidence_level=0.95, random_state=42)
                    ci_lower, ci_upper = res.confidence_interval
                    mean_val = np.mean(action_data)
                    # Calculate symmetric error bar size (average of distances to upper and lower bounds)
                    ci = (ci_upper - ci_lower) / 2
                else:
                    ci = 0
                day_confidence_intervals.append(ci)
            
            # Create bar plot for this day
            day_actions = list(day_action_rewards.index)
            day_avg_rewards = list(day_action_rewards.values)
            day_colors = [action_colors.get(action, 'gray') for action in day_actions]
            
            bars = ax.bar(day_actions, day_avg_rewards, color=day_colors, 
                         yerr=day_confidence_intervals, capsize=3, 
                         error_kw={'elinewidth': 1, 'capthick': 1})
            
            # Add sample size labels
            for j, (action, bar) in enumerate(zip(day_actions, bars)):
                sample_size = day_action_counts[action]
                bar_height = bar.get_height()
                ci = day_confidence_intervals[j] if j < len(day_confidence_intervals) else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar_height + ci + 0.005, 
                       f'n={sample_size}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'Day {day.day}', fontsize=10)
        ax.set_xlabel('Action', fontsize=9)
        ax.set_ylabel('Avg Reward', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        
        # Set consistent y-axis limits across all subplots
        all_rewards = reward_df_filtered['reward'].values
        y_min = np.min(all_rewards) - 0.02
        y_max = np.max(all_rewards) + 0.05
        ax.set_ylim(y_min, y_max)

# Hide unused subplots
for i in range(n_days, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Average Daily Change in Area by Action - Split by Day\n(with 95% Bootstrapped Confidence Intervals)', 
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("plots/average_reward_by_action_by_day_E11.png", dpi=300, bbox_inches="tight")

# Create plot grouped by first 6 days vs last 6 days
unique_days = sorted(reward_df_filtered['day'].unique())
n_total_days = len(unique_days)
print(f"Total unique days: {n_total_days}")

# Split into first 6 and last 6 days
first_6_days = unique_days[:6]
last_6_days = unique_days[-4:]

# Create data groups
first_6_data = reward_df_filtered[reward_df_filtered['day'].isin(first_6_days)]
last_6_data = reward_df_filtered[reward_df_filtered['day'].isin(last_6_days)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot for first 6 days
if not first_6_data.empty:
    first_action_rewards = first_6_data.groupby('action')['reward'].mean()
    first_action_counts = first_6_data.groupby('action').size()
    
    first_confidence_intervals = []
    for action in first_action_rewards.index:
        action_data = first_6_data[first_6_data['action'] == action]['reward'].values
        if len(action_data) > 1:
            def mean_statistic(x):
                return np.mean(x)
            
            res = bootstrap((action_data,), mean_statistic, n_resamples=5000, 
                           confidence_level=0.95, random_state=42)
            ci_lower, ci_upper = res.confidence_interval
            ci = (ci_upper - ci_lower) / 2
        else:
            ci = 0
        first_confidence_intervals.append(ci)
    
    first_actions = list(first_action_rewards.index)
    first_avg_rewards = list(first_action_rewards.values)
    first_colors = [action_colors.get(action, 'gray') for action in first_actions]
    
    bars1 = ax1.bar(first_actions, first_avg_rewards, color=first_colors, 
                   yerr=first_confidence_intervals, capsize=5, 
                   error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add sample size labels
    for i, (action, bar) in enumerate(zip(first_actions, bars1)):
        sample_size = first_action_counts[action]
        bar_height = bar.get_height()
        ci = first_confidence_intervals[i]
        ax1.text(bar.get_x() + bar.get_width()/2, bar_height + ci + 0.005, 
                f'n={sample_size}', ha='center', va='bottom', fontsize=10)

ax1.set_title(f'First 6 Days', fontsize=12)
ax1.set_xlabel('Action', fontsize=11)
ax1.set_ylabel('Average Reward', fontsize=11)

# Plot for last 6 days
if not last_6_data.empty:
    last_action_rewards = last_6_data.groupby('action')['reward'].mean()
    last_action_counts = last_6_data.groupby('action').size()
    
    last_confidence_intervals = []
    for action in last_action_rewards.index:
        action_data = last_6_data[last_6_data['action'] == action]['reward'].values
        if len(action_data) > 1:
            def mean_statistic(x):
                return np.mean(x)
            
            res = bootstrap((action_data,), mean_statistic, n_resamples=5000, 
                           confidence_level=0.95, random_state=42)
            ci_lower, ci_upper = res.confidence_interval
            ci = (ci_upper - ci_lower) / 2
        else:
            ci = 0
        last_confidence_intervals.append(ci)
    
    last_actions = list(last_action_rewards.index)
    last_avg_rewards = list(last_action_rewards.values)
    last_colors = [action_colors.get(action, 'gray') for action in last_actions]
    
    bars2 = ax2.bar(last_actions, last_avg_rewards, color=last_colors, 
                   yerr=last_confidence_intervals, capsize=5, 
                   error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add sample size labels
    for i, (action, bar) in enumerate(zip(last_actions, bars2)):
        sample_size = last_action_counts[action]
        bar_height = bar.get_height()
        ci = last_confidence_intervals[i]
        ax2.text(bar.get_x() + bar.get_width()/2, bar_height + ci + 0.005, 
                f'n={sample_size}', ha='center', va='bottom', fontsize=10)

ax2.set_title(f'Last 4 days', fontsize=12)
ax2.set_xlabel('Action', fontsize=11)
ax2.set_ylabel('Average Reward', fontsize=11)

# Set consistent y-axis limits across both subplots
all_rewards = reward_df_filtered['reward'].values
y_min = np.min(all_rewards)
y_max = np.max(all_rewards) + 0.05
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

plt.suptitle('Average Daily Change in Area by Action - First vs Last 6 Days\n(with 95% Bootstrapped Confidence Intervals)', 
             fontsize=14)
plt.tight_layout()
plt.savefig("plots/average_reward_by_action_first_vs_last_6_days_E11.png", dpi=300, bbox_inches="tight")

# plt.show()

# Initialize reward column as float to avoid dtype mismatch
# df["reward"] = 0.0  # Initialize reward column to 0.0
x = 1.0


# # %%
# # Set reward to 0 for all entries except the last of the day for each agent
# for agent, group in df.groupby("agent"):
#     # print(agent)
#     daily_groups = group.groupby("day")
#     for day, daily_group in daily_groups:
#         last_obs_index = daily_group.index[
#             -1
#         ]  # Get index of the last observation of the day
#         reward_row = reward_df.loc[
#             (reward_df["agent"] == agent) & (reward_df["day"] == day)
#         ]
#         if not reward_row.empty:  # Check if reward_row is not empty
#             df.loc[last_obs_index, "reward"] = reward_row["reward"].values[0]

# # Group data by day and calculate reward and percentage of agent_action == 1
# returns = {}
# for (day_idx, agent), group in df.groupby(["day_idx", "agent"]):  # type: ignore
#     if day_idx in list(range(3, 13)):
#         reward = group["reward"].max()  # Get the reward for the last observation
#         if agent in returns:
#             returns[agent] += reward
#         else:
#             returns[agent] = reward




# # Manually define unique colors for agents, using names from ZONE_TO_AGENT_E7 and ZONE_TO_AGENT_E8
# manual_colors = {
#     "Bernoulli p=0.90": "pink",
#     "Bernoulli p=0.85": "cyan",
#     "Bernoulli p=0.70": "gold",
#     "Bernoulli p=0.65": "lime",
# }

# # Create a bar plot for returns
# plt.figure(figsize=(10, 6))

# # Extract agents and their corresponding returns
# agents = list(returns.keys())
# agent_returns = [returns[agent] for agent in agents]

# # Use the colors from manual_colors
# colors = [
#     manual_colors.get(agent, "gray") for agent in agents
# ]  # Default to 'gray' if agent not in manual_colors

# # Plot the bar chart
# plt.bar(agents, agent_returns, color=colors)

# # Add title and labels
# plt.title(
#     "Undiscounted Return by Agent - Day 4 through Bolting \n (reward = %% change between avg of first 5 obs each day)",
#     fontsize=14,
# )
# plt.xlabel("Agent", fontsize=12)
# plt.ylabel("Return", fontsize=12)

# # Rotate x-axis labels for better readability
# plt.xticks(fontsize=10)

# # Save the plot
# plt.savefig("plots/outputs/agent_returns_percent.png", dpi=300, bbox_inches="tight")

# Create a new plot with subplots for each zone showing time vs mean_clean_area
# %%
# zones = sorted(df['zone'].unique())
# n_zones = len(zones)

# # Create color mapping for agent actions
# action_colors = {'White': 'yellow', 'Blue': 'blue', 'Red': 'red', 'Night': 'black'}

# # Calculate subplot layout (2 columns)
# n_cols = 2
# n_rows = (n_zones + n_cols - 1) // n_cols  # Ceiling division

# # Create subplots with 2 columns
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
# if n_rows == 1 and n_cols == 1:
#     axes = [[axes]]  # Make it a 2D array for consistency
# elif n_rows == 1 or n_cols == 1:
#     axes = axes.reshape(n_rows, n_cols)

# for i, zone in enumerate(zones):
#     row = i // n_cols
#     col = i % n_cols
#     ax = axes[row, col]
    
#     zone_data = df[df['zone'] == zone].sort_values('time')
    
#     # Plot scatter points colored by action (no connecting lines)
#     for action in zone_data['agent_action'].unique():
#         action_data = zone_data[zone_data['agent_action'] == action]
#         color = action_colors.get(action, 'gray')
#         ax.scatter(action_data['time'], action_data['mean_clean_area'], 
#                   c=color, alpha=0.7, s=2, label=action)
    
#     ax.set_title(f'Zone {zone}: Mean Area over Time')
#     ax.set_ylabel('Mean Area (normalized)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Format x-axis to show dates nicely
#     ax.tick_params(axis='x', rotation=45)

# # Hide any unused subplots
# for i in range(n_zones, n_rows * n_cols):
#     row = i // n_cols
#     col = i % n_cols
#     axes[row, col].set_visible(False)

# # Set common x-axis label for bottom row
# for col in range(n_cols):
#     if n_rows > 0:
#         axes[n_rows-1, col].set_xlabel('Time')

# plt.tight_layout()
# plt.savefig("/workspaces/plant-rl-oliver/plots/mean_clean_area_by_zone_E11.png", dpi=300, bbox_inches="tight")

# # %%
# # Create a new plot showing mean_clean_area at 9:30 each morning, colored by action at 9:40
# zones = sorted(df['zone'].unique())
# n_zones = len(zones)

# # Calculate subplot layout (2 columns)
# n_cols = 2
# n_rows = (n_zones + n_cols - 1) // n_cols  # Ceiling division

# # Create subplots with 2 columns
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
# if n_rows == 1 and n_cols == 1:
#     axes = [[axes]]  # Make it a 2D array for consistency
# elif n_rows == 1 or n_cols == 1:
#     axes = axes.reshape(n_rows, n_cols)

# for i, zone in enumerate(zones):
#     row = i // n_cols
#     col = i % n_cols
#     ax = axes[row, col]
    
#     zone_data = df[df['zone'] == zone].sort_values('time')
    
#     # Filter for 9:30 AM observations
#     morning_data = zone_data[zone_data['time'].dt.strftime('%H:%M') == '09:30'].copy()
    
#     # For each 9:30 observation, find the corresponding 9:40 action
#     morning_data['action_color'] = 'gray'  # default color
#     for idx in morning_data.index:
#         obs_time = zone_data.loc[idx, 'time']
#         # Look for 9:40 observation on the same day
#         same_day_data = zone_data[zone_data['time'].dt.date == obs_time.date()]
#         action_940 = same_day_data[same_day_data['time'].dt.strftime('%H:%M') == '09:40']
#         if not action_940.empty:
#             action = action_940['agent_action'].iloc[0]
#             morning_data.loc[idx, 'action_color'] = action_colors.get(action, 'gray')
    
#     # Plot connected line in black
#     ax.plot(morning_data['time'], morning_data['mean_clean_area'], 
#            color='black', alpha=0.7, linewidth=1, zorder=1)
    
#     # Plot colored dots based on 9:40 action
#     for action in morning_data['action_color'].unique():
#         action_points = morning_data[morning_data['action_color'] == action]
#         if not action_points.empty:
#             # Find the original action name for legend
#             action_name = [k for k, v in action_colors.items() if v == action]
#             action_name = action_name[0] if action_name else 'Unknown'
#             ax.scatter(action_points['time'], action_points['mean_clean_area'], 
#                       c=action, s=30, alpha=0.8, label=f'9:40 Action: {action_name}', zorder=2)
    
#     ax.set_title(f'Zone {zone}: Mean Clean Area at 9:30 AM (colored by 9:40 action)')
#     ax.set_ylabel('Mean Clean Area (normalized)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Format x-axis to show dates nicely
#     ax.tick_params(axis='x', rotation=45)

# # Hide any unused subplots
# for i in range(n_zones, n_rows * n_cols):
#     row = i // n_cols
#     col = i % n_cols
#     axes[row, col].set_visible(False)

# # Set common x-axis label for bottom row
# for col in range(n_cols):
#     if n_rows > 0:
#         axes[n_rows-1, col].set_xlabel('Time')

# plt.tight_layout()
# plt.savefig("/workspaces/plant-rl-oliver/plots/mean_clean_area_930_E11.png", dpi=300, bbox_inches="tight")
# plt.show()

# # %%

