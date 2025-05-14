#%%
import pandas as pd

df_1 = pd.read_csv("/data/online/E7/P2/Constant1/z1/raw.csv")

df_2 = pd.read_csv("/data/online/E7/P2/Constant2/z2/raw.csv")

# Convert time columns to datetime
df_1['time'] = pd.to_datetime(df_1['time'])
df_2['time'] = pd.to_datetime(df_2['time'])

# Convert time to America/Edmonton timezone
df_1['time'] = df_1['time'].dt.tz_convert('America/Edmonton')
df_2['time'] = df_2['time'].dt.tz_convert('America/Edmonton')

#%%
df_1.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator, HourLocator

# Add a source column to identify each dataset
df_1['source'] = 'Constant Dim'
df_2['source'] = 'Constant Standard'

# Combine the dataframes for easier plotting
combined_df = pd.concat([df_1, df_2], ignore_index=True)

# Set up the plot with Seaborn styling
plt.figure(figsize=(12, 6))
sns.set_style('whitegrid')

# Create the timeseries plot with datetime x-axis
sns.lineplot(data=df_1, x='time', y='mean_clean_area', label='Constant Dim')
sns.lineplot(data=df_2, x='time', y='mean_clean_area', label='Constant Standard')

# Format the x-axis to show dates nicely
ax = plt.gca()

# Set up locators for ticks at day boundaries and at 9am/9pm
days = DayLocator(tz='America/Edmonton')
hours = HourLocator(byhour=[9, 21], tz='America/Edmonton')  # 9am and 9pm

# Format the date labels
date_format = DateFormatter('%Y-%m-%d', tz='America/Edmonton')
time_format = DateFormatter('%H:%M', tz='America/Edmonton')

# Apply the locators
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_minor_formatter(time_format)

# Rotate labels for better readability
plt.xticks(rotation=90, ha='right')
plt.xticks(rotation=90, ha='right', minor=True)

# Add title and labels
plt.title('Mean Clean Area Over Time', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Mean Clean Area', fontsize=12)
plt.legend(title='Data Source')

# Improve the appearance
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

#%%
# Create a shifted version of df_2 with time shifted forward by 1 day
df_2_shifted = df_2.copy()
df_2_shifted['time'] = df_2_shifted['time'] + pd.Timedelta(days=1)
# Ensure timezone is preserved in the shifted dataframe
if df_2_shifted['time'].dt.tz is None:
    df_2_shifted['time'] = df_2_shifted['time'].dt.tz_localize('America/Edmonton')
elif df_2_shifted['time'].dt.tz.zone != 'America/Edmonton':
    df_2_shifted['time'] = df_2_shifted['time'].dt.tz_convert('America/Edmonton')

df_2_shifted['source'] = 'Constant Standard (shifted +1 day)'

# Set up the time-shifted comparison plot
plt.figure(figsize=(12, 6))
sns.set_style('whitegrid')

# Plot original df_1 and shifted df_2
sns.lineplot(data=df_1, x='time', y='mean_clean_area', label='Constant Dim', legend=False)
sns.lineplot(data=df_2_shifted, x='time', y='mean_clean_area', label='Constant Standard (shifted +1 day)', legend=False)

# Format the x-axis to show dates nicely
ax = plt.gca()
days = DayLocator(tz='America/Edmonton')
hours = HourLocator(byhour=[9, 21], tz='America/Edmonton')
date_format = DateFormatter('%Y-%m-%d', tz='America/Edmonton')
time_format = DateFormatter('%H:%M', tz='America/Edmonton')
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_minor_formatter(time_format)
plt.xticks(rotation=90, ha='right')
plt.xticks(rotation=90, ha='right', minor=True)

# Add title and labels
plt.title('Mean Clean Area Over Time (Constant Standard Shifted +1 Day)', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Mean Clean Area', fontsize=12)
plt.annotate('Constant Dim', xy=(0, 0.1), xycoords='axes fraction', fontsize=12, color=ax.get_lines()[0].get_color())
plt.annotate('Constant Standard\n(shifted +1 day)', xy=(0.8, 0.9), xycoords='axes fraction', fontsize=12, color=ax.get_lines()[1].get_color())

# remove splines
sns.despine()

# Improve the appearance
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# %%
# Plot reward timeseries for the shifted timeseries

# Set up the reward time-shifted comparison plot
plt.figure(figsize=(12, 6))
sns.set_style('whitegrid')

# Plot rewards from original df_1 and shifted df_2
sns.lineplot(data=df_1, x='time', y='reward', label='Constant Dim', alpha=0.8, legend=False)
sns.lineplot(data=df_2_shifted, x='time', y='reward', label='Constant Standard (shifted +1 day)', alpha=0.8, legend=False)

# Format the x-axis to show dates nicely
ax = plt.gca()
days = DayLocator(tz='America/Edmonton')
hours = HourLocator(byhour=[9, 21], tz='America/Edmonton')
date_format = DateFormatter('%Y-%m-%d', tz='America/Edmonton')
time_format = DateFormatter('%H:%M', tz='America/Edmonton')
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_minor_formatter(time_format)
plt.xticks(rotation=90, ha='right')
plt.xticks(rotation=90, ha='right', minor=True)

# Add title and labels
plt.title('Rewards Over Time (Constant Standard Shifted +1 Day)', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.annotate('Constant Dim', xy=(0, 0.1), xycoords='axes fraction', fontsize=12, color=ax.get_lines()[0].get_color())
plt.annotate('Constant Standard\n(shifted +1 day)', xy=(0.8, 0.9), xycoords='axes fraction', fontsize=12, color=ax.get_lines()[1].get_color())

# remove splines
sns.despine()

# Improve the appearance
plt.tight_layout()
plt.grid(True)

# Show the plot
plt.show()
# %%
