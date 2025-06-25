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
targets_12 = ['11:40', '11:50', '12:00', '12:10', '12:20']
targets_9 = ['09:20', '09:30', '09:40', '09:50', '10:00']  # Target times for 12pm observations
reward_data = []
returns = {agent: {} for agent in df['agent'].unique()}  # Initialize returns for each agent
for agent, group in df.groupby('agent'):
    group = group.sort_values('time')
    daily_groups = list(group.groupby('day'))  # Convert to list for indexing

    for i in range(len(daily_groups) - 1):  # Iterate up to the second last day
        current_day, current_group = daily_groups[i]
        next_day, next_group = daily_groups[i + 1]
        day_idx = current_group['day_idx'].iloc[0]
        print(f"Processing agent: {agent}, current day: {current_day}, next day: {next_day}")
        # Check if the days are consecutive
        if (next_day - current_day).days == 1:
            print(f"Found consecutive days: {current_day} and {next_day}")
            # Average of 5 obs centered around 12pm
            current_12pm_obs = current_group[current_group['time'].dt.strftime('%H:%M').isin(targets_12)].drop_duplicates(subset=['time'])
            next_12pm_obs = next_group[next_group['time'].dt.strftime('%H:%M').isin(targets_12)].drop_duplicates(subset=['time'])
            print(f"Number of observations in current_12pm_obs: {len(current_12pm_obs)}")
            print(f"Number of observations in next_12pm_obs: {len(next_12pm_obs)}")
            has_12pm_obs = not current_12pm_obs.empty and not next_12pm_obs.empty
            
            current_9am_obs = current_group[current_group['time'].dt.strftime('%H:%M').isin(targets_9)].drop_duplicates(subset=['time'])
            next_9am_obs = next_group[next_group['time'].dt.strftime('%H:%M').isin(targets_9)].drop_duplicates(subset=['time'])
            print(f"Number of observations in current_9am_obs: {len(current_9am_obs)}")
            print(f"Number of observations in next_9am_obs: {len(next_9am_obs)}")
            has_9am_obs = not current_9am_obs.empty and not next_9am_obs.empty
            
            # Max 5 obs
            current_max_obs = current_group.drop_duplicates(subset=['time']).nlargest(5, 'mean_clean_area')
            next_max_obs = next_group.drop_duplicates(subset=['time']).nlargest(5, 'mean_clean_area')
            print(f"Current day max observations at times {current_max_obs['time'].dt.strftime('%H:%M').values}:")
            print(f"Values: {current_max_obs['mean_clean_area'].values}")
            print(f"Next day max observations at times {next_max_obs['time'].dt.strftime('%H:%M').values}:")
            print(f"Values: {next_max_obs['mean_clean_area'].values}")
            has_max_obs = not current_max_obs.empty and not next_max_obs.empty

            if has_max_obs and has_12pm_obs and has_9am_obs:
                current_max = current_max_obs['mean_clean_area'].mean()
                next_max = next_max_obs['mean_clean_area'].mean()
                reward_max_raw = (next_max - current_max)
                reward_max_pct = (next_max - current_max) / current_max
                
                current_12pm_mean = current_12pm_obs['mean_clean_area'].mean()
                next_12pm_mean = next_12pm_obs['mean_clean_area'].mean()
                reward_12_raw = (next_12pm_mean - current_12pm_mean)
                reward_12_pct = (next_12pm_mean - current_12pm_mean) / current_12pm_mean
                
                current_9am_mean = current_9am_obs['mean_clean_area'].mean()
                next_9am_mean = next_9am_obs['mean_clean_area'].mean()
                reward_9_raw = (next_9am_mean - current_9am_mean)
                reward_9_pct = (next_9am_mean - current_9am_mean) / current_9am_mean
                
                reward_data.append({'agent': agent, 'day': current_day, 'day_idx': day_idx, 'reward_max_raw': reward_max_raw, 'reward_12_raw': reward_12_raw, 'reward_9_raw': reward_9_raw, 'reward_max_pct': reward_max_pct, 'reward_12_pct': reward_12_pct, 'reward_9_pct': reward_9_pct})
                    
                    # for rew in ['reward_max_raw', 'reward_12_raw', 'reward_9_raw', 'reward_max_pct', 'reward_12_pct', 'reward_9_pct']:
                    #         returns[agent][rew] = returns[agent].get(rew, 0) + rew
            else:
                print(f"Skipping agent {agent} for days {current_day} and {next_day} due to missing observations.")
                continue

reward_df = pd.DataFrame(reward_data)

returns = {
    agent: {
        col: reward_df[(reward_df['agent'] == agent) & (reward_df['day_idx'].isin(range(3, 12)))][col].sum()
        for col in reward_df.columns if col.startswith('reward')
    }
    for agent in reward_df['agent'].unique()
}
# Manually define unique colors for agents, using names from ZONE_TO_AGENT_E7 and ZONE_TO_AGENT_E8
manual_colors = {
    "Bernoulli p=0.90": 'pink',
    "Bernoulli p=0.85": 'cyan',
    "Bernoulli p=0.70": 'gold',
    "Bernoulli p=0.65": 'lime'
}

# Plot each return as a subplot with colors corresponding to each agent and save
# Create one figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()  # Flatten axes array for easier iteration

for idx, rew in enumerate(['reward_max_raw', 'reward_12_raw', 'reward_9_raw', 
                          'reward_max_pct', 'reward_12_pct', 'reward_9_pct']):
    ax = axes[idx]

    # Extract agents and their corresponding returns
    agents = list(returns.keys())
    agent_returns = [returns[agent].get(rew, 0) for agent in agents]

    # Use the colors from manual_colors
    colors = [manual_colors.get(agent, 'gray') for agent in agents]

    # Plot the bar chart
    ax.bar([agent_name.lstrip("Bernoulli ") for agent_name in returns.keys()], agent_returns, color=colors)

    #ax.set_xlabel('Agents', fontsize=14)
    ax.set_ylabel('Return', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    # ax.tick_params(axis='x', rotation=45)

    if rew.endswith('_raw'):
        desc = 'Reward = Raw change between\n'
    else:
        desc = 'Reward = % change between\n'

    if rew.startswith('reward_max'):
        desc += 'max 5 obs each day'
    elif rew.startswith('reward_12'):
        desc += 'avg of 5 obs centered around 12pm each day'
    elif rew.startswith('reward_9'):
        desc += 'avg of first 5 obs each day'

    ax.set_title(desc, fontsize=16)

plt.suptitle("Undiscounted Returns for Bernoulli Agents - 4 days after incubation through day 12 (bolting)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
plt.savefig('plots/outputs/returns.png', bbox_inches='tight')
plt.close()
