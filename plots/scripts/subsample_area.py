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
df.head(5)

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

#%%
# # Bootstrap analysis functions
# def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
#     """
#     Compute bootstrap confidence interval for the mean.
#     """
#     if len(data) == 0:
#         return np.nan, np.nan
    
#     bootstrap_means = []
#     for _ in range(n_bootstrap):
#         sample = np.random.choice(data, size=len(data), replace=True)
#         bootstrap_means.append(np.mean(sample))
    
#     alpha = 1 - confidence
#     lower_percentile = (alpha / 2) * 100
#     upper_percentile = (1 - alpha / 2) * 100
    
#     ci_lower = np.percentile(bootstrap_means, lower_percentile)
#     ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
#     return ci_lower, ci_upper

# def cis_overlap(ci1, ci2):
#     """
#     Check if two confidence intervals overlap.
#     """
#     return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])

def subsample_and_compute_ci_overlap(df, n_plants, timestep_interval=12, agents_to_analyze=None, k_iterations=500):
    """
    Perform bootstrap analysis for CI overlap between agents.
    
    Args:
        df: DataFrame with columns ['time', 'area', 'plant_id', 'agent']
        n_plants: Number of plants to subsample per agent
        timestep_interval: Analyze every Nth timestep
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
    
    # Get unique time steps - subsample every Nth timestep
    all_time_steps = sorted(df['time'].unique())
    time_steps = all_time_steps[::timestep_interval]  # Every Nth timestep
    
    # Pre-group data by agent and plant for efficiency
    print("Pre-grouping data by agent and plant...")
    grouped_data = {}
    for agent in agents:
        agent_data = df[df['agent'] == agent]
        grouped_data[agent] = {}
        
        # Create a pivot table: rows = plants, columns = timesteps
        agent_pivot = agent_data.pivot_table(
            index='plant_id', 
            columns='time', 
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
        print(f"Agent {agent}: {len(agent_plants[agent])} plants available")
    
    # Initialize results dictionary
    results = {pair: {time_step: [] for time_step in time_steps} for pair in agent_pairs}
    
    # Create random number generator for reproducibility
    rng = np.random.default_rng(42)
    
    for iteration in tqdm(range(k_iterations), desc="Bootstrap iterations"):
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
                iterations=1000
            )
            
            (ci_lower, ci_upper) = res.ci
            
            # Store CIs for each timestep (all timesteps are guaranteed to be in time_steps)
            agent_cis[agent] = dict(zip(plant_data.columns, zip(ci_lower, ci_upper)))
        
        # Check overlap for each pair at each timestep (fully vectorized)
        for pair in agent_pairs:
            agent1, agent2 = pair
            
            # Get CI dictionaries for both agents
            cis1 = agent_cis[agent1]
            cis2 = agent_cis[agent2]
            
            # Find timesteps where both agents have data
            common_timesteps = [ts for ts in cis1.keys() if ts in cis2]
            
            if common_timesteps:
                # Extract all CI bounds for common timesteps in vectorized fashion
                ci1_bounds = np.array([cis1[ts] for ts in common_timesteps])  # Shape: (n_timesteps, 2)
                ci2_bounds = np.array([cis2[ts] for ts in common_timesteps])  # Shape: (n_timesteps, 2)
                
                ci1_lower, ci1_upper = ci1_bounds[:, 0], ci1_bounds[:, 1]
                ci2_lower, ci2_upper = ci2_bounds[:, 0], ci2_bounds[:, 1]
                
                # Vectorized overlap check: CIs don't overlap if ci1_upper < ci2_lower OR ci2_upper < ci1_lower
                no_overlap = (ci1_upper < ci2_lower) | (ci2_upper < ci1_lower)
                
                # Store results for all common timesteps
                for i, ts in enumerate(common_timesteps):
                    results[pair][ts].append(no_overlap[i])
    
    return results

def compute_proportion_no_overlap(results):
    """
    Compute proportion of runs where CIs didn't overlap for each pair and time step.
    """
    proportions = {}
    
    for pair, time_data in results.items():
        proportions[pair] = {}
        for time_step, overlap_list in time_data.items():
            if len(overlap_list) > 0:
                proportions[pair][time_step] = np.mean(overlap_list)
            else:
                proportions[pair][time_step] = np.nan
    
    return proportions

#%%
# Main analysis
print("Starting bootstrap analysis...")
print(f"Total timesteps available: {len(sorted(df['time'].unique()))}")

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
sample_sizes = [4, 8, 18, 36]
k_iterations = 500
timestep_interval = 12  # Analyze every Nth timestep
# Specify which agents to include in analysis (None = all agents)
agents_to_analyze = None  # e.g., ["Bernoulli p=0.90", "Bernoulli p=0.85"] or None for all

print(f"Analyzing every {timestep_interval}th timestep: {len(sorted(df['time'].unique())[::timestep_interval])} timesteps")

# Store results for all sample sizes
all_results = {}

for n_plants in tqdm(sample_sizes, desc="Sample sizes"):
    print(f"\nAnalyzing with n={n_plants} plants per agent...")
    
    # Run bootstrap analysis
    results = subsample_and_compute_ci_overlap(df, n_plants, timestep_interval, agents_to_analyze, k_iterations)
    
    # Compute proportions
    proportions = compute_proportion_no_overlap(results)
    
    all_results[n_plants] = proportions

print("Bootstrap analysis complete!")

#%%
# Create plots
# Use the same agent filtering for plotting
all_agents = df['agent'].unique()
if agents_to_analyze is not None:
    agents = [agent for agent in all_agents if agent in agents_to_analyze]
else:
    agents = all_agents

agent_pairs = list(itertools.combinations(agents, 2))
all_time_steps = sorted(df['time'].unique())
time_steps = all_time_steps[::timestep_interval]  # Every Nth timestep

# Convert time steps to numeric values for plotting (use index)
time_indices = list(range(len(time_steps)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

colors = ['blue', 'red', 'green', 'orange']
sample_size_labels = [f'n={n}' for n in sample_sizes]

for i, pair in enumerate(agent_pairs):
    ax = axes[i]
    
    for j, n_plants in enumerate(sample_sizes):
        proportions = all_results[n_plants][pair]
        
        # Extract proportions for each time step, using NaN for missing data
        y_values = [proportions.get(time_step, np.nan) for time_step in time_steps]
        
        ax.plot(time_indices, y_values, 
                color=colors[j], 
                label=sample_size_labels[j], 
                linewidth=2)
    
    ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step Index')
    ax.set_ylabel('Proportion of Runs\nwith Non-overlapping CIs')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)

# Adjust layout and save
plt.tight_layout()
plt.savefig('/workspaces/plant-rl-oliver/plots/bootstrap_ci_analysis.png', 
            dpi=300, bbox_inches='tight')

print(f"Plot saved to /workspaces/plant-rl-oliver/plots/bootstrap_ci_analysis.png")

#%%
# Print summary statistics
print("\nSummary Statistics:")
print("=" * 50)

for n_plants in sample_sizes:
    print(f"\nSample size: {n_plants} plants per agent")
    print("-" * 30)
    
    for pair in agent_pairs:
        proportions = all_results[n_plants][pair]
        prop_values = [v for v in proportions.values() if not np.isnan(v)]
        
        if prop_values:
            mean_prop = np.mean(prop_values)
            std_prop = np.std(prop_values)
            print(f"{pair[0]} vs {pair[1]}: "
                  f"Mean={mean_prop:.3f}, Std={std_prop:.3f}")
        else:
            print(f"{pair[0]} vs {pair[1]}: No valid data")

