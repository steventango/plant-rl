#%%
from pathlib import Path
from datetime import datetime, timedelta
import pytz

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevent X server requirement (useful when running headless or via SSH)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from datetime import time

ZONE_TO_AGENT_E7 = {
    "z1": "Constant Dim",
    "z2": "Constant Bright",
    "z3": "Bandit",
    "z6": "Contextual Bandit",
    "z8": "ESARSA_alpha=0.1",
    "z9": "ESARSA_alpha=0.25"
}

ZONE_TO_AGENT_E8 = {
    "z2": "Bernoulli p=0.90",
    "z3": "Bernoulli p=0.85",
    "z8": "Bernoulli p=0.70",
    "z9": "Bernoulli p=0.65",
}


dfs = []

datasets = []
for p in ['P1']:
    paths = Path("/data/online/E8").joinpath(p).glob("Bernoulli*/z*")
    datasets.extend(sorted(paths))
for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    zone = dataset.name  # gets last directory name (e.g., 'z1', 'z2', etc.)
    df["agent"] = ZONE_TO_AGENT_E8[zone]  # convert zone to agent name
    df = df[["time", "agent_action", "mean_clean_area", "agent"]]  # keep only needed columns
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
    edmonton_tz = pytz.timezone('America/Edmonton')
    return utc_dt.astimezone(edmonton_tz)

df['time'] = df['time'].apply(convert_to_edmonton_timezone)

# Extract day from time column based on America/Edmonton timezone
df['day'] = df['time'].dt.date

# Create day_idx column for each agent group
for agent, group in df.groupby('agent'):
    # Sort by time to ensure correct day ordering
    sorted_group = group.sort_values('time')
    # Print unique times for each agent
    #print(f"\nAgent: {agent}")
    #print(sorted_group['time'].dt.strftime('%H:%M').unique())
    # Get unique days and create a mapping to indices
    first_day = min(sorted_group['day'])
    # Calculate day_idx as the number of days since the first day
    df.loc[sorted_group.index, 'day_idx'] = (sorted_group['day'] - first_day).apply(lambda x: x.days)


# # Normalize mean_clean_area
# for agent, group in df.groupby('agent'):
#     first_5_times = group[group['time'].dt.strftime('%H:%M').isin(['11:40', '11:50', '12:00', '12:10', '12:20'])].sort_values('time')['time'].unique()[:5]
#     #print(f"Agent: {agent}, First times: {first_5_times}")
#     first_obs_values = [group[group['time'] == t]['mean_clean_area'].iloc[0] for t in first_5_times]  # Get corresponding values
#     for t in list(zip(first_5_times, first_obs_values)):
#         print(f"Agent: {agent}, Time: {t[0]}, Value: {t[1]}")
#     first_obs_mean = sum(first_obs_values) / len(first_obs_values)  # Calculate mean of those values
#     df.loc[group.index, 'mean_clean_area'] = group['mean_clean_area'] / first_obs_mean

    #%%

# target_times = ['11:40', '11:50', '12:00', '12:10', '12:20']  # Target times 
target_times = ['09:20', '09:30', '09:40', '09:50', '10:00']  # Target times for 12pm observations
reward_data = []
for agent, group in df.groupby('agent'):
    group = group.sort_values('time')
    daily_groups = list(group.groupby('day'))  # Convert to list for indexing

    for i in range(len(daily_groups) - 1):  # Iterate up to the second last day
        current_day, current_group = daily_groups[i]
        next_day, next_group = daily_groups[i + 1]
        print(f"Processing agent: {agent}, current day: {current_day}, next day: {next_day}")
        # Check if the days are consecutive
        if (next_day - current_day).days == 1:
            print(f"Found consecutive days: {current_day} and {next_day}")
            # Filter observations for target times and drop duplicates
            current_9am_obs = current_group[current_group['time'].dt.strftime('%H:%M').isin(target_times)].drop_duplicates(subset=['time'])
            next_9am_obs = next_group[next_group['time'].dt.strftime('%H:%M').isin(target_times)].drop_duplicates(subset=['time'])
            print(f"Number of observations in current_9am_obs: {len(current_9am_obs)}")
            print(f"Number of observations in next_9am_obs: {len(next_9am_obs)}")

            # Ensure both days have observations at 12pm
            if not current_9am_obs.empty and not next_9am_obs.empty:
                current_9am_mean = current_9am_obs['mean_clean_area'].mean()
                next_9am_mean = next_9am_obs['mean_clean_area'].mean()

                diff = (next_9am_mean - current_9am_mean) / current_9am_mean
                reward_data.append({'agent': agent, 'day': current_day, 'reward': diff})

reward_df = pd.DataFrame(reward_data)
# Initialize reward column as float to avoid dtype mismatch
df['reward'] = 0.0  # Initialize reward column to 0.0


#%%
# Set reward to 0 for all entries except the last of the day for each agent
for agent, group in df.groupby('agent'):
    #print(agent)
    daily_groups = group.groupby('day')
    for day, daily_group in daily_groups:
        last_obs_index = daily_group.index[-1]  # Get index of the last observation of the day
        reward_row = reward_df.loc[(reward_df['agent'] == agent) & (reward_df['day'] == day)]
        if not reward_row.empty:  # Check if reward_row is not empty
            df.loc[last_obs_index, 'reward'] = reward_row['reward'].values[0]

# Group data by day and calculate reward and percentage of agent_action == 1
returns = {}
for (day_idx, agent), group in df.groupby(['day_idx', 'agent']):
    if day_idx in list(range(3, 13)):
        reward = group['reward'].max()  # Get the reward for the last observation 
        if agent in returns:
            returns[agent] += reward
        else:
            returns[agent] = reward

# Manually define unique colors for agents, using names from ZONE_TO_AGENT_E7 and ZONE_TO_AGENT_E8
manual_colors = {
            "Bernoulli p=0.90": 'pink',
            "Bernoulli p=0.85": 'cyan',
            "Bernoulli p=0.70": 'gold',
            "Bernoulli p=0.65": 'lime'
        }

# Create a bar plot for returns
plt.figure(figsize=(10, 6))

# Extract agents and their corresponding returns
agents = list(returns.keys())
agent_returns = [returns[agent] for agent in agents]

# Use the colors from manual_colors
colors = [manual_colors.get(agent, 'gray') for agent in agents]  # Default to 'gray' if agent not in manual_colors

# Plot the bar chart
plt.bar(agents, agent_returns, color=colors)

# Add title and labels
plt.title("Undiscounted Return by Agent - Day 4 through Bolting \n (reward = %% change between avg of first 5 obs each day)", fontsize=14)
plt.xlabel("Agent", fontsize=12)
plt.ylabel("Return", fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=10)

# Save the plot
plt.savefig("plots/outputs/agent_returns_percent.png", dpi=300, bbox_inches='tight')