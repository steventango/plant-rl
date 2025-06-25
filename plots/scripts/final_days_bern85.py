#%%
from pathlib import Path
from datetime import datetime, timedelta
import pytz

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevent X server requirement (useful when running headless or via SSH)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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

datasets = []
for p in ['P2', 'P3', 'P4']:
    paths = Path("/data/online/E7").joinpath(p).glob("**/z*")
    datasets.extend(sorted(paths))

for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    zone = dataset.name  # gets last directory name (e.g., 'z1', 'z2', etc.)
    df["agent"] = ZONE_TO_AGENT_E7[zone]  # convert zone to agent name
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


# Normalize mean_clean_area
for agent, group in df.groupby('agent'):
    first_5_times = group[group['time'].dt.strftime('%H:%M').isin(['11:40', '11:50', '12:00', '12:10', '12:20'])].sort_values('time')['time'].unique()[:5]
    #print(f"Agent: {agent}, First times: {first_5_times}")
    first_obs_values = [group[group['time'] == t]['mean_clean_area'].iloc[0] for t in first_5_times]  # Get corresponding values
    for t in list(zip(first_5_times, first_obs_values, strict=False)):
        print(f"Agent: {agent}, Time: {t[0]}, Value: {t[1]}")
    first_obs_mean = sum(first_obs_values) / len(first_obs_values)  # Calculate mean of those values
    df.loc[group.index, 'mean_clean_area'] = group['mean_clean_area'] / first_obs_mean

# Filter for unique values of time and the "Constant Bright" agent
bernoulli85_data = df[df['agent'] == "Bernoulli p=0.85"].drop_duplicates(subset=['time'])

# Filter for the final 3 days according to day_idx
# max_day_idx = bernoulli85_data['day_idx'].max()
# final_days_data = bernoulli85_data[bernoulli85_data['day_idx'] >= max_day_idx - 2]

# Filter for days 13, 14, and 15
final_days_data = bernoulli85_data[bernoulli85_data['day_idx'].isin([10, 11, 12, 13])]

final_days_data['new_day'] = final_days_data['time'].dt.strftime('%H:%M').isin(['09:20'])
# Plot time vs mean_clean_area for the final 3 days
plt.figure(figsize=(12, 6))
plt.plot(final_days_data['time'], final_days_data['mean_clean_area'])

# Force buffer to allow text visibility
#ymin, ymax = plt.ylim()
#plt.ylim(ymin, ymax + 0.2)

plt.title("Normalized Area for 'Bernoulli p=0.85' Agent (Days 10, 11, 12, 13)", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Normalized Area", fontsize=12)
# Set major ticks to be every hour
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
# Format ticks as day-hour-minute
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%H-%M', tz=pytz.timezone('America/Edmonton')))
plt.xticks(rotation=45)

# Extract unique times and corresponding day indices for annotation
new_day_rows = final_days_data[final_days_data['new_day']][['time', 'day_idx']]

for _, row in new_day_rows.iterrows():
    t = row['time']
    day = int(row['day_idx'])
    plt.axvline(x=t, color='red', linestyle='--')
    plt.text(
        t - timedelta(hours=0.5),   # shift left by 30 minutes
        final_days_data['mean_clean_area'].max() -0.05,
        f"Day {day}",
        color='red',
        rotation=0,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=10
    )


noon_rows = final_days_data[final_days_data['time'].dt.strftime('%H:%M') == '12:00']
for t in noon_rows['time']:
    plt.axvline(x=t, color='purple', linestyle=':', linewidth=1)
    plt.text(
        t + pd.Timedelta(hours=0.5),
        final_days_data['mean_clean_area'].max() - 0.05,
        '12pm',
        color='purple',
        fontsize=10,
        rotation=0,
        verticalalignment='bottom',
        horizontalalignment='left'
    )

plt.savefig("plots/outputs/bernoulli85.png", dpi=300)

