# %%
from datetime import datetime
from pathlib import Path
import itertools
import numpy as np
from scipy import stats
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytz

matplotlib.use(
    "Agg"
)  # Prevent X server requirement (useful when running headless or via SSH)

from RlEvaluation.temporal import curve_percentile_bootstrap_ci
from diff_bootstrap import curve_percentile_bootstrap_ci_diff

ZONE_TO_AGENT_E8 = {
    "z2": "Bernoulli p=0.90",
    "z3": "Bernoulli p=0.85",
    "z8": "Bernoulli p=0.70",
    "z9": "Bernoulli p=0.65",
}

AGENT_TO_PERCENT_E8 = {
    "Bernoulli p=0.90": 0.90,
    "Bernoulli p=0.85": 0.85,
    "Bernoulli p=0.70": 0.70,
    "Bernoulli p=0.65": 0.65,
}

# Set random seed for reproducibility
np.random.seed(42)

#============= PARAMETERS ============
#=====================================
total_days = 14  # Total days to analyze
sample_sizes = [8, 12, 18, 24, 30, 36]
k_iterations = 1000
n_bootstrap = 5000
timestep_interval = 6  # Analyze every Nth timestep (ignored if specific_times is used)
specific_times = ["09:20", "13:20", "17:20"]  # Specific times to analyze, or None to use interval
agents_to_analyze = None #  # e.g., ["Bernoulli p=0.90", "Bernoulli p=0.85"] or None for all
#=====================================
#=====================================

dfs = []

datasets = []
paths = Path("../data").glob("Bernoulli*/z*")
datasets.extend(sorted(paths))

for dataset in datasets:
    csv_path = dataset / "raw.csv"
    df = pd.read_csv(csv_path)
    # for col in df.columns:
    #     print(f"Column: {col}")
    #     print(f"First 3 elements: {df[col].head(3).tolist()}")
    #     print("-" * 40)
    zone = dataset.name  # gets last directory name (e.g., 'z1', 'z2', etc.)
    df["agent"] = ZONE_TO_AGENT_E8[zone]  # convert zone to agent name
    df = df[
        ["time", "area", 'plant_id', "agent"]
    ]  # keep only needed columns
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


#%%
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
# Round timestamps to the nearest minute to handle fractional second differences
df["time_rounded"] = df["time"].dt.floor('min')

# Filter to only the first 14 days of data
earliest_date = df["time_rounded"].min().date()
cutoff_date = earliest_date + pd.Timedelta(days=total_days)
df = df[df["time_rounded"].dt.date < cutoff_date]
print(f"Filtered to first 14 days: {earliest_date} to {cutoff_date - pd.Timedelta(days=1)}")

#%%

def subsample_and_compute_ci_overlap(df, n_plants, timestep_interval=12, specific_times=None, agents_to_analyze=None, k_iterations=500, n_bootstrap=10000):
    """
    Perform bootstrap analysis for CI overlap between agents.
    
    Args:
        df: DataFrame with columns ['time', 'area', 'plant_id', 'agent']
        n_plants: Number of plants to subsample per agent
        timestep_interval: Analyze every Nth timestep (ignored if specific_times is provided)
        specific_times: List of specific times to analyze in "HH:MM" format (e.g., ["10:00", "14:30"])
        agents_to_analyze: List of agent names to include, or None for all agents
        k_iterations: Number of bootstrap iterations
    
    Returns:
        Dictionary with results for each agent pair and time step
    """
    # Filter agents if specified
    all_agents = df['agent'].unique()
    if agents_to_analyze is not None:
        agents = [agent for agent in all_agents if agent in agents_to_analyze]
        df = df[df['agent'].isin(agents)]
        print(f"Analyzing subset of agents: {agents}")
    else:
        agents = all_agents
        print(f"Analyzing all agents: {agents}")
    
    agent_pairs = list(itertools.combinations(agents, 2))
    
    # Get unique time steps - either specific times or every Nth timestep
    all_time_steps = sorted(df['time_rounded'].unique())
    
    if specific_times is not None:
        # Filter to only timesteps that match the specified times of day
        time_steps = []
        for timestamp in all_time_steps:
            time_str = timestamp.strftime("%H:%M")
            if time_str in specific_times:
                time_steps.append(timestamp)
        print(f"Using specific times {specific_times}: found {len(time_steps)} matching timesteps")
    else:
        # Use interval-based selection
        time_steps = all_time_steps[::timestep_interval]  # Every Nth timestep
        print(f"Using interval {timestep_interval}: analyzing {len(time_steps)} timesteps")
    
    # Pre-group data by agent and plant for efficiency
    print("Pre-grouping data by agent and plant...")
    grouped_data = {}
    for agent in agents:
        agent_data = df[df['agent'] == agent]
        grouped_data[agent] = {}
        
        # Create a pivot table: rows = plants, columns = timesteps
        agent_pivot = agent_data.pivot_table(
            index='plant_id', 
            columns='time_rounded', 
            values='area', 
            aggfunc='first'
        )
        
        # Filter to only include the timesteps we want (every Nth)
        available_timesteps = [ts for ts in time_steps if ts in agent_pivot.columns]
        agent_pivot_filtered = agent_pivot[available_timesteps]
        
        grouped_data[agent] = agent_pivot_filtered
        print(f"Agent {agent}: {len(agent_pivot_filtered)} plants, {len(available_timesteps)} timesteps")
    
    # Get available plants for each agent
    agent_plants = {}
    for agent in agents:
        agent_plants[agent] = grouped_data[agent].index.values
    
    # Check if we're using all plants for all agents (optimization for deterministic case)
    using_all_plants = all(len(agent_plants[agent]) <= n_plants for agent in agents)
    if using_all_plants:
        print(f"Using all available plants ({n_plants} requested). Only 1 iteration needed.")
        actual_k_iterations = 1
    else:
        actual_k_iterations = k_iterations
    
    # Initialize results dictionaries
    results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    mean_diff_results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    # Store sample means from each iteration for std error computation
    sample_means_results = {agent: {time_step: [] for time_step in time_steps} for agent in agents}
    # Store reward values (change from 1 day earlier divided by 3rd value in sample_stat)
    reward_results = {agent: {time_step: [] for time_step in time_steps} for agent in agents}
    # Store whether higher percentage agent had larger mean
    higher_percent_larger_mean_results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    # Store whether higher percentage agent had larger reward
    higher_percent_larger_reward_results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    
    # Create random number generator for reproducibility
    rng = np.random.default_rng(42)
    
    for iteration in tqdm(range(actual_k_iterations), desc="Bootstrap iterations"):
        # For this iteration, subsample plants for each agent
        sampled_plants = {}
        for agent in agents:
            available_plants = agent_plants[agent]
            
            # Check if we have enough plants
            if len(available_plants) < n_plants:
                if iteration == 0:  # Only warn once
                    print(f"Warning: Agent {agent} has only {len(available_plants)} plants, but {n_plants} requested")
                sampled_plants[agent] = available_plants  # Use all available plants
            else:
                sampled_plants[agent] = rng.choice(available_plants, size=n_plants, replace=False)
        
        # For each agent, compute CIs for all timesteps at once
        agent_cis = {}
        for agent in agents:
            # Get data for sampled plants (rows = plants, cols = timesteps)
            plant_data = grouped_data[agent].loc[sampled_plants[agent]]
            
            # Convert to numpy array (plants x timesteps)
            data_array = plant_data.values
            
            # Use curve_percentile_bootstrap_ci to get CIs for all timesteps
            res = curve_percentile_bootstrap_ci(
                rng=rng,
                y=data_array,
                alpha=0.05,  # 95% confidence interval
                iterations=n_bootstrap
            )
            
            (ci_lower, ci_upper) = res.ci
            
            # Store CIs for each timestep
            agent_cis[agent] = dict(zip(plant_data.columns, zip(ci_lower, ci_upper)))
            
            # Calculate reward: change from 1 day earlier divided by 3rd value in sample_stat
            timestamps = list(plant_data.columns)
            rewards = np.full(len(res.sample_stat), np.nan)  # Initialize with NaN
            
            # Find timesteps that are 1 day earlier for vectorized calculation
            for i, current_ts in enumerate(timestamps):
                # Look for timestamp 1 day earlier
                target_ts = current_ts - pd.Timedelta(days=1)
                if target_ts in timestamps:
                    earlier_idx = timestamps.index(target_ts)
                    # Calculate reward: (current_area - earlier_area) / third_value
                    if len(res.sample_stat) > 2:  # Ensure we have at least 3 values
                        starting_area = res.sample_stat[2]
                        rewards[i] = (res.sample_stat[i] - res.sample_stat[earlier_idx]) / starting_area
            
            # Store sample means for each timestep (res.sample_stat contains the means for this iteration)
            for i, ts in enumerate(plant_data.columns):
                sample_means_results[agent][ts].append(res.sample_stat[i])
                reward_results[agent][ts].append(rewards[i])
        
        # Check overlap and compute mean differences for each pair at each timestep
        for pair in agent_pairs:
            agent1, agent2 = pair
            
            # Get data for both agents (rows = plants, cols = timesteps)
            plant_data1 = grouped_data[agent1].loc[sampled_plants[agent1]]
            plant_data2 = grouped_data[agent2].loc[sampled_plants[agent2]]
            
            # Find common timesteps between the two agents
            common_timesteps = [ts for ts in plant_data1.columns if ts in plant_data2.columns]
            
            if common_timesteps:
                # Filter to common timesteps and convert to numpy arrays
                data_array1 = plant_data1[common_timesteps].values  # (n_plants1, n_timesteps)
                data_array2 = plant_data2[common_timesteps].values  # (n_plants2, n_timesteps)
                
                # Get CI dictionaries for both agents (for overlap check)
                cis1 = {ts: agent_cis[agent1][ts] for ts in common_timesteps}
                cis2 = {ts: agent_cis[agent2][ts] for ts in common_timesteps}
                
                # Extract all CI bounds for common timesteps in vectorized fashion
                ci1_bounds = np.array([cis1[ts] for ts in common_timesteps])  # Shape: (n_timesteps, 2)
                ci2_bounds = np.array([cis2[ts] for ts in common_timesteps])  # Shape: (n_timesteps, 2)
                
                ci1_lower, ci1_upper = ci1_bounds[:, 0], ci1_bounds[:, 1]
                ci2_lower, ci2_upper = ci2_bounds[:, 0], ci2_bounds[:, 1]
                
                # Vectorized overlap check: CIs don't overlap if ci1_upper < ci2_lower OR ci2_upper < ci1_lower
                no_overlap = (ci1_upper < ci2_lower) | (ci2_upper < ci1_lower)
                
                # Use your new function to compute CIs for the difference in means
                diff_res = curve_percentile_bootstrap_ci_diff(
                    rng=rng,
                    x=data_array1,  # (n_plants1, n_timesteps)
                    y=data_array2,  # (n_plants2, n_timesteps)
                    alpha=0.05,  # 95% confidence interval
                    iterations=n_bootstrap
                )
                
                # diff_res.sample_stat should be the observed difference in means for each timestep
                # diff_res.ci should be the (lower, upper) bounds for each timestep
                mean_diffs = diff_res.sample_stat  # Shape: (n_timesteps,)
                diff_ci_lower, diff_ci_upper = diff_res.ci  # Each shape: (n_timesteps,)
                
                # Check if CI excludes zero (significant difference)
                ci_excludes_zero = (diff_ci_lower > 0) | (diff_ci_upper < 0)  # Shape: (n_timesteps,)
                
                # Store results for all common timesteps
                for i, ts in enumerate(common_timesteps):
                    results[pair][ts].append(no_overlap[i])
                    # Store whether the CI for the difference excludes zero
                    mean_diff_results[pair][ts].append(ci_excludes_zero[i])
                    
                    # Check if higher percentage agent had larger mean
                    agent1_percent = AGENT_TO_PERCENT_E8[agent1]
                    agent2_percent = AGENT_TO_PERCENT_E8[agent2]
                    agent1_mean = sample_means_results[agent1][ts][-1]  # Get the last (current) mean
                    agent2_mean = sample_means_results[agent2][ts][-1]  # Get the last (current) mean
                    agent1_reward = reward_results[agent1][ts][-1]  # Get the last (current) reward
                    agent2_reward = reward_results[agent2][ts][-1]  # Get the last (current) reward
                    
                    if agent1_percent > agent2_percent:
                        # Agent1 has higher percentage, check if it had larger mean
                        higher_percent_larger_mean_results[pair][ts].append(agent1_mean > agent2_mean)
                        # Handle NaN rewards - only compare if both are not NaN
                        if not (np.isnan(agent1_reward) or np.isnan(agent2_reward)):
                            higher_percent_larger_reward_results[pair][ts].append(agent1_reward > agent2_reward)
                        else:
                            higher_percent_larger_reward_results[pair][ts].append(np.nan)
                    else:
                        # Agent2 has higher percentage, check if it had larger mean
                        higher_percent_larger_mean_results[pair][ts].append(agent2_mean > agent1_mean)
                        # Handle NaN rewards - only compare if both are not NaN
                        if not (np.isnan(agent1_reward) or np.isnan(agent2_reward)):
                            higher_percent_larger_reward_results[pair][ts].append(agent2_reward > agent1_reward)
                        else:
                            higher_percent_larger_reward_results[pair][ts].append(np.nan)

    # If we only did 1 iteration (using all plants), replicate the result k_iterations times
    # This ensures the proportion calculation works correctly
    if actual_k_iterations == 1:
        print(f"Replicating single deterministic result {k_iterations} times for proportion calculation...")
        for pair in agent_pairs:
            for ts in time_steps:
                if len(results[pair][ts]) == 1:
                    # Replicate the single result k_iterations times
                    single_result = results[pair][ts][0]
                    single_mean_diff = mean_diff_results[pair][ts][0]
                    single_higher_percent_mean = higher_percent_larger_mean_results[pair][ts][0]
                    single_higher_percent_reward = higher_percent_larger_reward_results[pair][ts][0]
                    results[pair][ts] = [single_result] * k_iterations
                    mean_diff_results[pair][ts] = [single_mean_diff] * k_iterations
                    higher_percent_larger_mean_results[pair][ts] = [single_higher_percent_mean] * k_iterations
                    higher_percent_larger_reward_results[pair][ts] = [single_higher_percent_reward] * k_iterations
        
        # Replicate standard error and reward results
        for agent in agents:
            for ts in time_steps:
                if len(sample_means_results[agent][ts]) == 1:
                    single_sample_mean = sample_means_results[agent][ts][0]
                    single_reward = reward_results[agent][ts][0]
                    sample_means_results[agent][ts] = [single_sample_mean] * k_iterations
                    reward_results[agent][ts] = [single_reward] * k_iterations
    
    # Compute standard errors from the collected sample means
    std_error_results = {agent: {} for agent in agents}
    for agent in agents:
        for ts in time_steps:
            if len(sample_means_results[agent][ts]) > 1:
                # Standard error is the standard deviation of sample means across iterations
                std_error_results[agent][ts] = np.std(sample_means_results[agent][ts])
            elif len(sample_means_results[agent][ts]) == 1:
                # If only one value, std error is 0 (deterministic case)
                std_error_results[agent][ts] = 0.0
            else:
                std_error_results[agent][ts] = np.nan
    
    return results, mean_diff_results, std_error_results, higher_percent_larger_mean_results, higher_percent_larger_reward_results

def compute_proportion(results):
    """
    Compute proportion of runs where condition was True for each pair and time step.
    Works for any results dictionary with boolean values.
    """
    proportions = {}
    
    for pair, time_data in results.items():
        proportions[pair] = {}
        for time_step, boolean_list in time_data.items():
            if len(boolean_list) > 0:
                proportions[pair][time_step] = np.mean(boolean_list)
            else:
                proportions[pair][time_step] = np.nan
    
    return proportions

#%%
# Main analysis
print("Starting bootstrap analysis...")
print(f"Total timesteps available: {len(sorted(df['time_rounded'].unique()))}")

# Continue with analysis
if specific_times is not None:
    print(f"Analyzing specific times {specific_times}")
else:
    print(f"Analyzing every {timestep_interval}th timestep: {len(sorted(df['time_rounded'].unique())[::timestep_interval])} timesteps")

# Store results for all sample sizes
all_results = {}
all_mean_diff_results = {}
all_std_error_results = {}
all_higher_percent_larger_mean_results = {}
all_higher_percent_larger_reward_results = {}

for n_plants in tqdm(sample_sizes, desc="Sample sizes"):
    print(f"\nAnalyzing with n={n_plants} plants per agent...")
    
    # Run bootstrap analysis
    results, mean_diff_results, std_error_results, higher_percent_larger_mean_results, higher_percent_larger_reward_results = subsample_and_compute_ci_overlap(df, n_plants, timestep_interval, specific_times, agents_to_analyze, k_iterations, n_bootstrap)
    
    # Compute proportions for CI overlap
    proportions = compute_proportion(results)
    
    # Compute proportions for significant mean differences
    mean_diff_proportions = compute_proportion(mean_diff_results)
    
    # Compute proportions for higher percentage agent having larger mean
    higher_percent_proportions = compute_proportion(higher_percent_larger_mean_results)
    
    # Compute proportions for higher percentage agent having larger reward
    higher_percent_reward_proportions = compute_proportion(higher_percent_larger_reward_results)
    
    all_results[n_plants] = proportions
    all_mean_diff_results[n_plants] = mean_diff_proportions
    all_std_error_results[n_plants] = std_error_results
    all_higher_percent_larger_mean_results[n_plants] = higher_percent_proportions
    all_higher_percent_larger_reward_results[n_plants] = higher_percent_reward_proportions

print("Bootstrap analysis complete!")

# Define agents and time_steps for standard error computation
all_agents = df['agent'].unique()
if agents_to_analyze is not None:
    agents = [agent for agent in all_agents if agent in agents_to_analyze]
else:
    agents = all_agents

all_time_steps = sorted(df['time_rounded'].unique())
if specific_times is not None:
    time_steps = []
    for timestamp in all_time_steps:
        time_str = timestamp.strftime("%H:%M")
        if time_str in specific_times:
            time_steps.append(timestamp)
else:
    time_steps = all_time_steps[::timestep_interval]

#%%
# Create plots for CI overlap
# Use the same agent filtering for plotting
all_agents = df['agent'].unique()
if agents_to_analyze is not None:
    agents = [agent for agent in all_agents if agent in agents_to_analyze]
else:
    agents = all_agents

agent_pairs = list(itertools.combinations(agents, 2))
all_time_steps = sorted(df['time_rounded'].unique())

# Use the same time selection logic as in the analysis
if specific_times is not None:
    time_steps = []
    for timestamp in all_time_steps:
        time_str = timestamp.strftime("%H:%M")
        if time_str in specific_times:
            time_steps.append(timestamp)
else:
    time_steps = all_time_steps[::timestep_interval]  # Every Nth timestep

# Convert time steps to day-hour-minute format for plotting
def format_timestamp_for_plot(timestamp):
    """Convert timestamp to day-hour-minute format relative to the first day."""
    earliest_date = df["time_rounded"].min()
    days_diff = (timestamp - earliest_date).days
    time_str = timestamp.strftime("%H:%M")
    return f"D{days_diff}-{time_str}"

time_labels = [format_timestamp_for_plot(ts) for ts in time_steps]
time_indices = list(range(len(time_steps)))

def create_plots(results_dict, title_suffix, filename_suffix):
    # Calculate dynamic subplot grid based on number of agent pairs
    n_pairs = len(agent_pairs)
    if n_pairs == 0:
        print("No agent pairs found for plotting")
        return
    elif n_pairs == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]  # Make it a list for consistent indexing
    elif n_pairs <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes = axes.flatten()
    elif n_pairs <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
    elif n_pairs <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        axes = axes.flatten()
    else:
        # For more than 6 pairs, use a 3-column layout
        n_rows = (n_pairs + 2) // 3  # Ceiling division
        fig, axes = plt.subplots(n_rows, 3, figsize=(24, 6 * n_rows))
        axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'orange']
    sample_size_labels = [f'n={n}' for n in sample_sizes]

    for i, pair in enumerate(agent_pairs):
        ax = axes[i]
        
        for j, n_plants in enumerate(sample_sizes):
            proportions = results_dict[n_plants][pair]
            
            # Extract proportions for each time step, using NaN for missing data
            y_values = [proportions.get(time_step, np.nan) for time_step in time_steps]
            
            ax.plot(time_indices, y_values, 
                    color=colors[j], 
                    label=sample_size_labels[j], 
                    linewidth=2, 
                    marker='o',
                    markersize=4,)
        
        ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Proportion of \n {title_suffix}')
        ax.set_xticks(time_indices)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)

    # Hide any unused subplots if we have more subplots than pairs
    if n_pairs > 0:
        for i in range(n_pairs, len(axes)):
            axes[i].set_visible(False)

    # Add main title for the entire grid
    fig.suptitle(f'Proportion of {title_suffix} ({k_iterations} trials per sample size)', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))  # Leave more space at bottom for rotated labels
    
    # Create descriptive filename with analysis parameters
    sample_sizes_str = "_".join(map(str, sample_sizes))
    if agents_to_analyze is not None and len(agents_to_analyze) > 0:
        agents_str = "_".join([agent.replace(" ", "").replace("=", "").replace(".", "") for agent in agents_to_analyze])
    else:
        agents_str = "all"
    
    if specific_times is not None:
        times_str = "_".join([t.replace(":", "") for t in specific_times])
        filename = f'bootstrap_{filename_suffix}_samples_{sample_sizes_str}_iters_{k_iterations}_times_{times_str}_agents_{agents_str}.png'
    else:
        filename = f'bootstrap_{filename_suffix}_samples_{sample_sizes_str}_iters_{k_iterations}_interval_{timestep_interval}_agents_{agents_str}.png'
    
    filepath = f'/workspaces/plant-rl/plots/{filename}'
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
    plt.close()

# Create both plots
create_plots(all_results, "Non-overlapping CIs", "ci_overlap")
create_plots(all_mean_diff_results, "Significant Mean Difference in Area", "mean_diff_significant")
create_plots(all_higher_percent_larger_mean_results, "Higher Percent Agent With Larger Mean Area", "higher_percent_larger")
create_plots(all_higher_percent_larger_reward_results, "Higher Percent Agent With Larger Reward", "higher_percent_larger_reward")

# Create standard error plot
def create_std_error_plot(std_error_results, sample_sizes, agents, time_steps, specific_times, timestep_interval, k_iterations, agents_to_analyze):
    """
    Plot standard errors across timesteps for different sample sizes and agents.
    """
    # Convert time steps to day-hour-minute format for plotting
    def format_timestamp_for_plot(timestamp):
        """Convert timestamp to day-hour-minute format relative to the first day."""
        earliest_date = df["time_rounded"].min()
        days_diff = (timestamp - earliest_date).days
        time_str = timestamp.strftime("%H:%M")
        return f"D{days_diff}-{time_str}"

    time_labels = [format_timestamp_for_plot(ts) for ts in time_steps]
    time_indices = list(range(len(time_steps)))
    
    # Calculate subplot grid based on number of agents
    n_agents = len(agents)
    if n_agents == 0:
        print("No agents found for plotting")
        return
    elif n_agents == 1:
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes = [axes]
    elif n_agents <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        axes = axes.flatten()
    elif n_agents <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
    else:
        # For more than 4 agents, use a 2-column layout
        n_rows = (n_agents + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))
        axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, agent in enumerate(agents):
        ax = axes[i]
        
        # Determine how many plants this agent has
        agent_data = df[df['agent'] == agent]
        n_plants_available = len(agent_data['plant_id'].unique())
        
        # Plot standard errors for each sample size (excluding those that use all plants)
        for j, n_plants in enumerate(sample_sizes):
            # Skip if this sample size uses all available plants for this agent
            if n_plants >= n_plants_available:
                continue
                
            if agent in std_error_results[n_plants]:
                # Get standard error for each timestep (now single values, not lists)
                std_errors = []
                for ts in time_steps:
                    if ts in std_error_results[n_plants][agent]:
                        std_errors.append(std_error_results[n_plants][agent][ts])
                    else:
                        std_errors.append(np.nan)
                
                # Only plot if we have valid standard errors
                if any(not np.isnan(se) for se in std_errors):
                    ax.plot(time_indices, std_errors, 
                            color=colors[j % len(colors)], 
                            label=f'n={n_plants}', 
                            linewidth=2, 
                            marker='o',
                            markersize=4)
        
        ax.set_title(f'{agent}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Standard Error')
        ax.set_xticks(time_indices)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable y-axis limits
        all_values = []
        for line in ax.lines:
            y_data = line.get_ydata()
            try:
                # Convert to numpy array and filter out NaN values
                y_array = np.asarray(y_data)
                valid_values = y_array[~np.isnan(y_array)]
                all_values.extend(valid_values.tolist())
            except:
                # Skip if there's any issue with the data
                continue
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)

    # Hide any unused subplots
    if n_agents > 0:
        for i in range(n_agents, len(axes)):
            axes[i].set_visible(False)

    # Add main title
    fig.suptitle(f'Standard Error of Mean Area ({k_iterations} trials per sample size)', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))  # Leave more space at bottom for rotated labels
    
    # Create filename
    sample_sizes_str = "_".join(map(str, sample_sizes))
    if agents_to_analyze is not None and len(agents_to_analyze) > 0:
        agents_str = "_".join([agent.replace(" ", "").replace("=", "").replace(".", "") for agent in agents_to_analyze])
    else:
        agents_str = "all"
    
    if specific_times is not None:
        times_str = "_".join([t.replace(":", "") for t in specific_times])
        filename = f'bootstrap_std_error_samples_{sample_sizes_str}_iters_{k_iterations}_times_{times_str}_agents_{agents_str}.png'
    else:
        filename = f'bootstrap_std_error_samples_{sample_sizes_str}_iters_{k_iterations}_interval_{timestep_interval}_agents_{agents_str}.png'
    
    filepath = f'/workspaces/plant-rl/plots/{filename}'
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Standard error plot saved to {filepath}")
    plt.close()

create_std_error_plot(all_std_error_results, sample_sizes, agents, time_steps, specific_times, timestep_interval, k_iterations, agents_to_analyze)

