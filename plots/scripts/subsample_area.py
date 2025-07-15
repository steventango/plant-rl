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
sample_sizes = [4, 8, 18, 28, 36]
k_iterations = 1000
n_bootstrap = 5000
timestep_interval = 6  # Analyze every Nth timestep (ignored if specific_times is used)
specific_times = ["09:20", "11:20", "13:20", "15:20", "17:20","19:20", "21:20"]  # Specific times to analyze, or None to use interval
return_calculation_time = "09:20"  # Specific time to use for daily return calculation
agents_to_analyze = None  # e.g., ["Bernoulli p=0.90", "Bernoulli p=0.85"] or None for all
total_plants = 36  # Total number of plants per agent (for subsampling)
threshold_for_correct_ranked_return = 0.95  # Threshold for percent agents correctly ranked in the return difference plot
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
def compute_true_rankings_all_plants(df, agents, time_steps):
    """
    Compute the "true" rankings using all available plants for each agent.
    
    Args:
        df: DataFrame with columns ['time', 'area', 'plant_id', 'agent']
        agents: List of agent names
        time_steps: List of timesteps to analyze
        
    Returns:
        true_mean_rankings: Dict with agent pairs as keys, timesteps as subkeys, and boolean values indicating if first agent has larger mean
        true_reward_rankings: Dict with agent pairs as keys, timesteps as subkeys, and boolean values indicating if first agent has larger reward
    """
    import itertools
    
    agent_pairs = list(itertools.combinations(agents, 2))
    
    # Pre-group data by agent for efficiency
    grouped_data = {}
    for agent in agents:
        agent_data = df[df['agent'] == agent]
        # Create a pivot table: rows = plants, columns = timesteps
        agent_pivot = agent_data.pivot_table(
            index='plant_id', 
            columns='time_rounded', 
            values='area', 
            aggfunc='first'
        )
        # Filter to only include the timesteps we want
        available_timesteps = [ts for ts in time_steps if ts in agent_pivot.columns]
        agent_pivot_filtered = agent_pivot[available_timesteps]
        grouped_data[agent] = agent_pivot_filtered
    
    # Compute true means and rewards for each agent at each timestep
    true_means = {agent: {} for agent in agents}
    true_rewards = {agent: {} for agent in agents}
    
    for agent in agents:
        plant_data = grouped_data[agent]
        timestamps = list(plant_data.columns)
        
        # Compute means for all timesteps
        means = plant_data.mean(axis=0)  # Mean across plants for each timestep
        
        # Store means
        for ts in timestamps:
            true_means[agent][ts] = means[ts]
        
        # Calculate rewards: change from 1 day earlier divided by 3rd value
        rewards = {}
        if len(means) > 2:
            starting_area = means.iloc[2]  # Third value as baseline
            for i, ts in enumerate(timestamps):
                # Look for timestamp 1 day earlier
                target_ts = ts - pd.Timedelta(days=1)
                if target_ts in timestamps:
                    earlier_idx = timestamps.index(target_ts)
                    reward = (means.iloc[i] - means.iloc[earlier_idx]) / starting_area
                    rewards[ts] = reward
                else:
                    rewards[ts] = np.nan
        
        true_rewards[agent] = rewards
    
    # Compute rankings for each pair
    true_mean_rankings = {pair: {} for pair in agent_pairs}
    true_reward_rankings = {pair: {} for pair in agent_pairs}
    
    for pair in agent_pairs:
        agent1, agent2 = pair
        
        # Find common timesteps
        common_timesteps = [ts for ts in time_steps 
                          if ts in true_means[agent1] and ts in true_means[agent2]]
        
        for ts in common_timesteps:
            # Mean ranking: True if agent1 has larger mean than agent2
            if not (np.isnan(true_means[agent1][ts]) or np.isnan(true_means[agent2][ts])):
                true_mean_rankings[pair][ts] = true_means[agent1][ts] > true_means[agent2][ts]
            else:
                true_mean_rankings[pair][ts] = np.nan
            
            # Reward ranking: True if agent1 has larger reward than agent2
            if (ts in true_rewards[agent1] and ts in true_rewards[agent2] and
                not (np.isnan(true_rewards[agent1][ts]) or np.isnan(true_rewards[agent2][ts]))):
                true_reward_rankings[pair][ts] = true_rewards[agent1][ts] > true_rewards[agent2][ts]
            else:
                true_reward_rankings[pair][ts] = np.nan
    
    return true_mean_rankings, true_reward_rankings

def subsample_and_compute_ci_overlap(df, n_plants, timestep_interval=12, specific_times=None, agents_to_analyze=None, k_iterations=500, n_bootstrap=10000, return_calculation_time="09:20"):
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
    
    # Validate return calculation time exists for each agent each day
    print(f"Validating return calculation time '{return_calculation_time}' exists for each agent each day...")
    return_time_valid = True
    all_time_steps_full = sorted(df['time_rounded'].unique())
    
    # Find all timestamps that match the return calculation time across all days
    return_timestamps = []
    for timestamp in all_time_steps_full:
        if timestamp.strftime("%H:%M") == return_calculation_time:
            return_timestamps.append(timestamp)
    
    expected_return_entries = len(return_timestamps)
    
    for agent in agents:
        agent_data = df[df['agent'] == agent]
        agent_return_entries = 0
        
        for timestamp in return_timestamps:
            # Check if this agent has data at this return calculation time
            agent_at_time = agent_data[agent_data['time_rounded'] == timestamp]
            if len(agent_at_time) > 0:
                agent_return_entries += 1
        
        if agent_return_entries != expected_return_entries:
            print(f"WARNING: Agent {agent} has {agent_return_entries} entries at {return_calculation_time}, expected {expected_return_entries}")
            return_time_valid = False
        else:
            print(f"Agent {agent}: {agent_return_entries}/{expected_return_entries} return calculation entries found")
    
    if not return_time_valid:
        print("WARNING: Not all agents have the same number of return calculation time entries!")
    else:
        print("All agents have valid return calculation time entries.")
    
    # Get available plants for each agent
    agent_plants = {}
    for agent in agents:
        agent_plants[agent] = grouped_data[agent].index.values
    
    # Compute true rankings using all available plants
    print("Computing true rankings using all available plants...")
    true_mean_rankings, true_reward_rankings = compute_true_rankings_all_plants(df, agents, time_steps)
    
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
    # Store whether higher percentage agent had larger return (sum of daily rewards)
    higher_percent_larger_return_results = {pair: [] for pair in agent_pairs}
    # Store agent returns for each agent for every iteration
    agent_returns_all_iterations = {agent: [] for agent in agents}
    # Store whether mean area ranking was correct (matches true ranking from all plants)
    correct_mean_ranking_results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    # Store whether reward ranking was correct (matches true ranking from all plants)
    correct_reward_ranking_results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    
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
                    
                    # Check if mean area ranking matches true ranking
                    if ts in true_mean_rankings[pair] and not np.isnan(true_mean_rankings[pair][ts]):
                        observed_mean_ranking = agent1_mean > agent2_mean  # True if agent1 > agent2
                        true_mean_ranking = true_mean_rankings[pair][ts]   # True if agent1 > agent2 in true data
                        correct_mean_ranking_results[pair][ts].append(observed_mean_ranking == true_mean_ranking)
                    else:
                        correct_mean_ranking_results[pair][ts].append(np.nan)
                    
                    # Check if reward ranking matches true ranking
                    if ts in true_reward_rankings[pair] and not np.isnan(true_reward_rankings[pair][ts]):
                        if not (np.isnan(agent1_reward) or np.isnan(agent2_reward)):
                            observed_reward_ranking = agent1_reward > agent2_reward  # True if agent1 > agent2
                            true_reward_ranking = true_reward_rankings[pair][ts]     # True if agent1 > agent2 in true data
                            correct_reward_ranking_results[pair][ts].append(observed_reward_ranking == true_reward_ranking)
                        else:
                            correct_reward_ranking_results[pair][ts].append(np.nan)
                    else:
                        correct_reward_ranking_results[pair][ts].append(np.nan)
        
        # Calculate agent returns (sum of daily rewards at return_calculation_time) for this iteration
        # This is done once per iteration, not per timestep
        agent_returns = {}
        for agent in agents:
            # Calculate return as sum of rewards at return_calculation_time across all days
            agent_return = 0
            valid_rewards_count = 0
            
            # Sum rewards for each day at the return_calculation_time
            for timestamp in return_timestamps:
                if timestamp in reward_results[agent]:
                    reward_vals = reward_results[agent][timestamp]
                    if reward_vals:  # If we have rewards recorded for this timestamp
                        current_reward = reward_vals[-1]  # Get the most recent reward for this iteration
                        if not np.isnan(current_reward):
                            agent_return += current_reward
                            valid_rewards_count += 1
            
            # Only store the return if we found at least one valid reward
            if valid_rewards_count > 0:
                agent_returns[agent] = agent_return
            else:
                agent_returns[agent] = np.nan
            
            # Store the agent return for this iteration
            agent_returns_all_iterations[agent].append(agent_returns[agent])
        
        agent_return_diffs = {}
        # Compare agent returns for each pair
        for pair in agent_pairs:
            agent1, agent2 = pair
            agent1_percent = AGENT_TO_PERCENT_E8[agent1]
            agent2_percent = AGENT_TO_PERCENT_E8[agent2]
            agent1_return = agent_returns.get(agent1, np.nan)
            agent2_return = agent_returns.get(agent2, np.nan)
            
            if not (np.isnan(agent1_return) or np.isnan(agent2_return)):
                if using_all_plants:
                    agent_return_diffs[pair] = abs(agent1_return - agent2_return)
                if agent1_percent > agent2_percent:
                    higher_percent_larger_return_results[pair].append(agent1_return > agent2_return)
                else:
                    higher_percent_larger_return_results[pair].append(agent2_return > agent1_return)
            else:
                higher_percent_larger_return_results[pair].append(np.nan)

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
                    single_correct_mean = correct_mean_ranking_results[pair][ts][0]
                    single_correct_reward = correct_reward_ranking_results[pair][ts][0]
                    results[pair][ts] = [single_result] * k_iterations
                    mean_diff_results[pair][ts] = [single_mean_diff] * k_iterations
                    higher_percent_larger_mean_results[pair][ts] = [single_higher_percent_mean] * k_iterations
                    higher_percent_larger_reward_results[pair][ts] = [single_higher_percent_reward] * k_iterations
                    correct_mean_ranking_results[pair][ts] = [single_correct_mean] * k_iterations
                    correct_reward_ranking_results[pair][ts] = [single_correct_reward] * k_iterations
        
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
    
    return results, mean_diff_results, std_error_results, higher_percent_larger_mean_results, higher_percent_larger_reward_results, higher_percent_larger_return_results, agent_returns_all_iterations, correct_mean_ranking_results, correct_reward_ranking_results

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

def compute_proportion_returns(results):
    """
    Compute proportion of runs where condition was True for each pair (return results don't have time dimension).
    Works for return results dictionary with boolean values.
    """
    proportions = {}
    
    for pair, boolean_list in results.items():
        if len(boolean_list) > 0:
            # Filter out NaN values before computing proportion
            valid_values = [val for val in boolean_list if not np.isnan(val)]
            if len(valid_values) > 0:
                proportions[pair] = np.mean(valid_values)
            else:
                proportions[pair] = np.nan
        else:
            proportions[pair] = np.nan
    
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
all_higher_percent_larger_return_results = {}
all_agent_returns = {}
all_correct_mean_ranking_results = {}
all_correct_reward_ranking_results = {}

for n_plants in tqdm(sample_sizes, desc="Sample sizes"):
    print(f"\nAnalyzing with n={n_plants} plants per agent...")
    
    # Run bootstrap analysis
    results, mean_diff_results, std_error_results, higher_percent_larger_mean_results, higher_percent_larger_reward_results, higher_percent_larger_return_results, agent_returns_all_iterations, correct_mean_ranking_results, correct_reward_ranking_results = subsample_and_compute_ci_overlap(df, n_plants, timestep_interval, specific_times, agents_to_analyze, k_iterations, n_bootstrap, return_calculation_time)
    
    # Compute proportions for CI overlap
    proportions = compute_proportion(results)
    
    # Compute proportions for significant mean differences
    mean_diff_proportions = compute_proportion(mean_diff_results)
    
    # Compute proportions for higher percentage agent having larger mean
    higher_percent_proportions = compute_proportion(higher_percent_larger_mean_results)
    
    # Compute proportions for higher percentage agent having larger reward
    higher_percent_reward_proportions = compute_proportion(higher_percent_larger_reward_results)
    
    # Compute proportions for higher percentage agent having larger return
    higher_percent_return_proportions = compute_proportion_returns(higher_percent_larger_return_results)
    
    # Compute proportions for correct mean area rankings
    correct_mean_ranking_proportions = compute_proportion(correct_mean_ranking_results)
    
    # Compute proportions for correct reward rankings
    correct_reward_ranking_proportions = compute_proportion(correct_reward_ranking_results)
    
    all_results[n_plants] = proportions
    all_mean_diff_results[n_plants] = mean_diff_proportions
    all_std_error_results[n_plants] = std_error_results
    all_higher_percent_larger_mean_results[n_plants] = higher_percent_proportions
    all_higher_percent_larger_reward_results[n_plants] = higher_percent_reward_proportions
    all_higher_percent_larger_return_results[n_plants] = higher_percent_return_proportions
    all_agent_returns[n_plants] = agent_returns_all_iterations
    all_correct_mean_ranking_results[n_plants] = correct_mean_ranking_proportions
    all_correct_reward_ranking_results[n_plants] = correct_reward_ranking_proportions

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

    # Generate colors dynamically based on number of sample sizes
    # Use a cycle of distinctive colors that will repeat if needed
    base_colors = ['blue', 'red', 'green', 'orange', 'black', 'brown', 'gray', 'pink', 'olive', 'cyan']
    colors = [base_colors[i % len(base_colors)] for i in range(len(sample_sizes))]
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

        xticks = [i for i, label in enumerate(time_labels) if label.endswith(specific_times[0])]
        xlabels = [time_labels[i] for i in xticks]
        
        ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Proportion of \n {title_suffix}')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
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


def create_return_plot(return_results):
    """
    Create a plot for return comparison results (no time dimension).
    x-axis: number of plants, y-axis: proportion of higher-percent agent having larger return
    Creates a subplot for each agent pair.
    """
    # Get sample sizes and agent pairs
    sample_sizes = sorted(return_results.keys())
    if not sample_sizes:
        print("No return results to plot")
        return
    
    # Get agent pairs from the first sample size
    agent_pairs = list(return_results[sample_sizes[0]].keys())
    if not agent_pairs:
        print("No agent pairs found in return results")
        return
    
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
    
    # Plot each agent pair in its own subplot
    for i, pair in enumerate(agent_pairs):
        ax = axes[i]
        
        proportions = []
        valid_sample_sizes = []
        
        for n_plants in sample_sizes:
            if pair in return_results[n_plants]:
                prop = return_results[n_plants][pair]
                if not np.isnan(prop):
                    proportions.append(prop)
                    valid_sample_sizes.append(n_plants)
        
        if proportions:  # Only plot if we have valid data
            # Format pair label for subplot title
            agent1, agent2 = pair
            agent1_percent = AGENT_TO_PERCENT_E8[agent1] * 100
            agent2_percent = AGENT_TO_PERCENT_E8[agent2] * 100
            
            ax.plot(valid_sample_sizes, proportions, 'o-', color='blue',
                   linewidth=2, markersize=8, alpha=0.8)
            
            # Customize each subplot
            ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Size', fontsize=10)
            ax.set_ylabel('Proportion of Trails with \n Larger Return for Higher % Agent', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis ticks to only show the actual sample sizes used
            ax.set_xticks(valid_sample_sizes)
            ax.set_xticklabels([str(n) for n in valid_sample_sizes])
            
            # Set y-axis limits with some padding
            ax.set_ylim(-0.05, 1.05)
    
    # Hide any unused subplots if we have more subplots than pairs
    if n_pairs > 0:
        for i in range(n_pairs, len(axes)):
            axes[i].set_visible(False)
    
    # Add main title for the entire grid
    fig.suptitle(f'Proportion of Trails with Larger Return for Higher Percent \nAgent ({k_iterations} trials per sample size)', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    # Create filename with consistent naming convention
    sample_sizes_str = "_".join(map(str, sample_sizes))
    if agents_to_analyze is not None and len(agents_to_analyze) > 0:
        agents_str = "_".join([agent.replace(" ", "").replace("=", "").replace(".", "") for agent in agents_to_analyze]) # type: ignore

    else:
        agents_str = "all"
    
    if specific_times is not None:
        times_str = "_".join([t.replace(":", "") for t in specific_times])
        filename = f'higher_percent_larger_return_samples_{sample_sizes_str}_iters_{k_iterations}_times_{times_str}_agents_{agents_str}.png'
    else:
        filename = f'higher_percent_larger_return_samples_{sample_sizes_str}_iters_{k_iterations}_interval_{timestep_interval}_agents_{agents_str}.png'
    
    filepath = f'/workspaces/plant-rl/plots/{filename}'
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Return comparison plot saved to {filepath}")
    plt.close()

# Create both plots
create_plots(all_results, "Non-overlapping CIs", "ci_overlap")
create_plots(all_mean_diff_results, "Significant Difference in Mean Area", "mean_diff_significant")
create_plots(all_higher_percent_larger_mean_results, "Higher Percent Agent With Larger Mean Area", "higher_percent_larger_area")
create_plots(all_higher_percent_larger_reward_results, "Higher Percent Agent With Larger Reward", "higher_percent_larger_reward")
create_plots(all_correct_mean_ranking_results, "Correctly Ranked Mean Area", "correct_mean_ranking")
create_plots(all_correct_reward_ranking_results, "Correctly Ranked Reward", "correct_reward_ranking")

# Create return comparison plot (different format since no time dimension)
create_return_plot(all_higher_percent_larger_return_results)

# Create standard error plot
def create_std_error_plot(std_error_results, sample_sizes, agents, time_steps, specific_times, timestep_interval, k_iterations, agents_to_analyze):
    """
    Plot standard errors across timesteps for different sample sizes and agents.
    """
    # Convert time steps to day-hour-minute format for plotting
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

    # Generate colors dynamically based on number of sample sizes
    base_colors = ['blue', 'red', 'green', 'orange', 'black', 'brown', 'gray', 'pink', 'olive', 'cyan']
    colors = [base_colors[i % len(base_colors)] for i in range(len(sample_sizes))]
    
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
                for idx, ts in enumerate(time_steps):
                    if ts in std_error_results[n_plants][agent]:
                        se = std_error_results[n_plants][agent][ts]
                        std_errors.append(se)
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
        
        xticks = [i for i, label in enumerate(time_labels) if label.endswith(specific_times[0])]
        xlabels = [time_labels[i] for i in xticks]
        
        ax.set_title(f'{agent}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Standard Error')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
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

# Create return difference vs sample size plot
def create_return_diff_vs_sample_size_plot(all_higher_percent_larger_return_results, all_agent_returns, sample_sizes, agents_to_analyze, total_plants, threshold=0.9):
    """
    Create a plot with x-axis as absolute value of difference between returns for each pair when n_plants=total_plants
    and y-axis as the smallest n_plants value such that the higher percent agent has >90% probability of larger return
    or <10% probability if the higher percent agent doesn't have larger return at n_plants=total_plants.
    """
    # Check if total_plants is in sample_sizes
    if total_plants not in sample_sizes:
        print("Warning: n_plants=total_plants not found in sample_sizes. Cannot create return difference plot.")
        return
    
    # Get agent pairs
    all_agents = df['agent'].unique()
    if agents_to_analyze is not None:
        agents = [agent for agent in all_agents if agent in agents_to_analyze]
    else:
        agents = all_agents
    
    agent_pairs = list(itertools.combinations(agents, 2))
    
    x_values = []  # Absolute difference in returns at n_plants=total_plants
    y_values = []  # Smallest n_plants for threshold
    pair_labels = []
    
    for pair in agent_pairs:
        agent1, agent2 = pair
        agent1_percent = AGENT_TO_PERCENT_E8[agent1]
        agent2_percent = AGENT_TO_PERCENT_E8[agent2]
        
        # Get returns for n_plants=total_plants
        if total_plants in all_agent_returns and agent1 in all_agent_returns[total_plants] and agent2 in all_agent_returns[total_plants]:
            agent1_returns_total_plants = all_agent_returns[total_plants][agent1]
            agent2_returns_total_plants = all_agent_returns[total_plants][agent2]
            
            # Calculate absolute difference (assuming single iteration, take first value)
            if len(agent1_returns_total_plants) > 0 and len(agent2_returns_total_plants) > 0:
                agent1_return_total_plants = agent1_returns_total_plants[0] if not np.isnan(agent1_returns_total_plants[0]) else np.nan
                agent2_return_total_plants = agent2_returns_total_plants[0] if not np.isnan(agent2_returns_total_plants[0]) else np.nan
                
                if not (np.isnan(agent1_return_total_plants) or np.isnan(agent2_return_total_plants)):
                    abs_diff = abs(agent1_return_total_plants - agent2_return_total_plants)
                    
                    # Determine which agent has higher percentage
                    higher_percent_has_larger_return_total_plants = (
                        (agent1_percent > agent2_percent and agent1_return_total_plants > agent2_return_total_plants) or
                        (agent2_percent > agent1_percent and agent2_return_total_plants > agent1_return_total_plants)
                    )
                    
                    # Find smallest n_plants where threshold is met
                    target_threshold = 0.9 if higher_percent_has_larger_return_total_plants else 0.1
                    smallest_n_plants = None
                    
                    for n_plants in sorted(sample_sizes):
                        if pair in all_higher_percent_larger_return_results[n_plants]:
                            proportion = all_higher_percent_larger_return_results[n_plants][pair]
                            if not np.isnan(proportion):
                                if higher_percent_has_larger_return_total_plants and proportion > threshold:
                                    smallest_n_plants = n_plants
                                    break
                                elif not higher_percent_has_larger_return_total_plants and proportion < (1 - threshold):
                                    smallest_n_plants = n_plants
                                    break
                    
                    if smallest_n_plants is not None:
                        x_values.append(abs_diff)
                        y_values.append(smallest_n_plants)
                        pair_labels.append(f"{pair[0]} vs {pair[1]}")
    
    if not x_values:
        print("No valid data points found for return difference plot.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors for each pair (assuming max 6 pairs)
    pair_colors = {
        0: 'blue',
        1: 'red', 
        2: 'green',
        3: 'orange',
        4: 'purple',
        5: 'brown'
    }
    
    # Scatter plot with different colors for each pair
    for i, (x, y, label) in enumerate(zip(x_values, y_values, pair_labels)):
        color = pair_colors.get(i, 'black')  # Default to black if more than 6 pairs
        ax.scatter(x, y, s=100, alpha=0.7, c=color, label=label)
    
    # Connect all points with a line (sorted by x-value for cleaner visualization)
    sorted_indices = sorted(range(len(x_values)), key=lambda i: x_values[i])
    sorted_x = [x_values[i] for i in sorted_indices]
    sorted_y = [y_values[i] for i in sorted_indices]
    ax.plot(sorted_x, sorted_y, 'k--', alpha=0.5, linewidth=1, zorder=0)
    
    ax.set_xlabel(f'True Absolute Difference in Returns (Using All {total_plants} Plants)', fontsize=12)
    ax.set_ylabel(f'Min Sample Size For {threshold*100}% Accuracy', fontsize=12)
    ax.set_title(f'True Difference in Returns vs Min Sample Size \n for {threshold*100}% Accuracy in Return Ranking with {k_iterations} Trials)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Set y-axis to show only actual sample sizes
    ax.set_yticks(sorted(sample_sizes))
    ax.set_yticklabels([str(n) for n in sorted(sample_sizes)])
    
    # Save the plot
    sample_sizes_str = "_".join(map(str, sample_sizes))
    if agents_to_analyze is not None and len(agents_to_analyze) > 0:
        agents_str = "_".join([agent.replace(" ", "").replace("=", "").replace(".", "") for agent in agents_to_analyze])
    else:
        agents_str = "all"
    
    filename = f'return_diff_vs_sample_size_samples_{sample_sizes_str}_iters_{k_iterations}_agents_{agents_str}.png'
    filepath = f'/workspaces/plant-rl/plots/{filename}'
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Return difference vs min sample size plot saved to {filepath}")
    plt.close()

# Create the new plot
create_return_diff_vs_sample_size_plot(all_higher_percent_larger_return_results, all_agent_returns, sample_sizes, agents_to_analyze, total_plants, threshold=threshold_for_correct_ranked_return)

