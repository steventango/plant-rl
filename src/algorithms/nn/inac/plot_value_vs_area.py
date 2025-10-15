#!/usr/bin/env python3
"""
Script to load a trained network from checkpoint and plot Q-values as a function of area.

Usage:
    python plot_value_vs_area.py --checkpoint_dir /path/to/checkpoint --output plot_value_vs_area.png
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import minari
import numpy as np
import pandas as pd
import seaborn as sns
from flax import nnx

from algorithms.nn.inac.agent.in_sample import ActorCritic


def load_states_from_dataset(dataset):
    """
    Load all observation states from the Minari dataset.

    Args:
        dataset: MinariDataset to load states from

    Returns:
        states: (N, state_dim) array of all observations
        areas: (N,) array of area values (first dimension of observation)
        actions: (N,) array of action indices (extracted from one-hot encoding in state)
    """
    all_obs = []

    for episode in dataset.iterate_episodes():
        # Get all observations except the last one (which doesn't have an action)
        all_obs.append(episode.observations[:-1])

    states = jnp.concatenate(all_obs, axis=0)
    areas = np.array(states[:, 0])  # First dimension is area (normalized)

    # Extract action from one-hot encoding (indices 2-4)
    action_onehot = np.array(states[:, 2:5])
    actions = np.argmax(action_onehot, axis=1)

    return states, areas, actions


def get_area_normalization_params():
    """
    Get area normalization parameters from the dataset generation code.
    These should match the values used in datasets/env.py MockEnv._get_observation()

    Returns:
        clean_area_min: Minimum clean area value
        clean_area_max: Maximum clean area value
    """
    # Load the offline dataset to get normalization parameters
    import polars as pl

    try:
        df = pl.read_parquet("/data/offline/cleaned_offline_dataset_daily.parquet")
        clean_area_min = df["clean_area"].min()
        clean_area_max = df["clean_area"].max()
        if clean_area_min is not None and clean_area_max is not None:
            area_min = float(clean_area_min)  # type: ignore[arg-type]
            area_max = float(clean_area_max)  # type: ignore[arg-type]
            print(f"Loaded normalization params: min={area_min:.3f}, max={area_max:.3f}")
            return area_min, area_max
        else:
            raise ValueError("Min/max values are None")
    except Exception as e:
        print(f"Warning: Could not load normalization params from parquet: {e}")
        print("Using default normalization parameters")
        # Fallback values if parquet is not available
        return 0.0, 1.0

def plot_value_vs_area(
    areas: np.ndarray,
    q_values_by_action: np.ndarray,
    output_path: Path,
    area_min: float,
    area_max: float,
    bin_size: float = 10.0,
):
    """
    Plot next plant area as a function of current plant area, with one line per action.
    Data is binned and 95% bootstrap confidence intervals are computed.

    Args:
        areas: Array of normalized area values from dataset [0, 1], shape (num_states,)
        q_values_by_action: Array of Q-values, shape (num_actions, num_states)
        output_path: Path to save the plot
        area_min: Minimum area value for unnormalization
        area_max: Maximum area value for unnormalization
        bin_size: Size of bins in mm² (default: 10.0)
    """
    action_names = ["Red", "White", "Blue"]
    action_colors = ["red", "gray", "blue"]

    # Unnormalize areas
    areas_unnorm = areas * (area_max - area_min) + area_min

    # Create a DataFrame for seaborn
    data_list = []
    for action_idx in range(3):
        # Get Q-values for this action across all states
        q_values_action = q_values_by_action[action_idx]

        # Calculate next plant area: Q-value * plant_area + plant_area
        # Q-value represents the growth rate, so next_area = current_area * (1 + growth_rate)
        next_areas_action = q_values_action * areas_unnorm + areas_unnorm

        # Create bins for current area
        area_bins = np.arange(areas_unnorm.min(), areas_unnorm.max() + bin_size, bin_size)
        bin_centers = area_bins[:-1] + bin_size / 2

        # Assign each data point to a bin
        bin_indices = np.digitize(areas_unnorm, area_bins) - 1

        # Create DataFrame entries
        for i in range(len(areas_unnorm)):
            if 0 <= bin_indices[i] < len(bin_centers):
                data_list.append({
                    'Current Area (mm²)': bin_centers[bin_indices[i]],
                    'Next Area (mm²)': next_areas_action[i],
                    'Action': action_names[action_idx]
                })

    df = pd.DataFrame(data_list)

    print(f"Created dataframe with {len(df)} rows")
    print(f"Bins: {bin_size} mm² bins")
    print(f"Data points per action:")
    for action in action_names:
        print(f"  {action}: {len(df[df['Action'] == action])}")

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot with seaborn lineplot (automatically computes bootstrap CI)
    sns.lineplot(
        data=df,
        x='Current Area (mm²)',
        y='Next Area (mm²)',
        hue='Action',
        palette={'Red': 'red', 'White': 'gray', 'Blue': 'blue'},
        errorbar=('ci', 95),  # 95% bootstrap confidence interval
        err_style='band',
        linewidth=2.5,
        ax=ax,
        n_boot=1000,  # Number of bootstrap samples
    )

    # Add identity line (y=x) for reference
    area_range = [areas_unnorm.min(), areas_unnorm.max()]
    ax.plot(area_range, area_range, 'k--', alpha=0.3, linewidth=1.5, label='No growth (y=x)')

    ax.set_xlabel("Current Plant Area (mm²)", fontsize=14)
    ax.set_ylabel("Next Plant Area (mm²)", fontsize=14)
    ax.set_title(f"Predicted Next Plant Area vs Current Plant Area by Action\n(Binned every {bin_size} mm² with 95% Bootstrap CI)", fontsize=16)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()
def main():
    parser = argparse.ArgumentParser(
        description="Load checkpoint and plot value as a function of area"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (contains 'parameters' subdirectory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plot_value_vs_area.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="plant-rl/discrete-v5",
        help="Dataset name to load observations from",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--state_dim", type=int, default=5, help="State dimension"
    )
    parser.add_argument(
        "--action_dim", type=int, default=3, help="Action dimension"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=256, help="Hidden units"
    )
    parser.add_argument(
        "--discrete_control", action="store_true", default=True, help="Use discrete control"
    )
    parser.add_argument(
        "--bin_size", type=float, default=10.0, help="Bin size in mm² for aggregating data (default: 10.0)"
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    parameters_dir = checkpoint_dir / "parameters"

    if not parameters_dir.exists():
        raise ValueError(f"Parameters directory not found: {parameters_dir}")

    # Load dataset to get area normalization parameters
    try:
        dataset = minari.load_dataset(args.dataset)
        print(f"Loaded dataset: {args.dataset}")
    except Exception as e:
        print(f"Warning: Could not load dataset {args.dataset}: {e}")
        print("Using default normalization parameters")
        dataset = None

    # Initialize random number generators
    rngs = nnx.Rngs(args.seed)

    # Create actor-critic network
    print("Creating actor-critic network...")
    actor_critic = ActorCritic(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_units=args.hidden_units,
        discrete_control=args.discrete_control,
        rngs=rngs,
    )

    # Create dummy optimizer (needed for loading, but won't be used)
    import optax

    from algorithms.nn.inac.agent.in_sample import Optimizers

    optimizers = Optimizers(
        pi=nnx.Optimizer(actor_critic.pi, optax.adam(1e-4), wrt=nnx.Param),
        q=nnx.Optimizer(actor_critic.q, optax.adam(1e-4), wrt=nnx.Param),
        value=nnx.Optimizer(actor_critic.value_net, optax.adam(1e-4), wrt=nnx.Param),
        beh_pi=nnx.Optimizer(actor_critic.beh_pi, optax.adam(1e-4), wrt=nnx.Param),
    )

    # Load checkpoint
    print(f"Loading checkpoint from {parameters_dir}...")
    try:
        # Split the current modules to get graphdefs and states for the checkpoint structure
        module_graphdef, module_state = nnx.split(actor_critic)
        optimizers_graphdef, optimizers_state = nnx.split(optimizers)

        # Create target structure for restoration
        target = {
            "module_graphdef": module_graphdef,
            "module_state": module_state,
            "optimizers_graphdef": optimizers_graphdef,
            "optimizers_state": optimizers_state,
        }

        # Load with target structure
        import orbax.checkpoint as ocp
        with ocp.StandardCheckpointer() as checkpointer:
            ckpt = checkpointer.restore(
                str(parameters_dir / "default"),
                target
            )

        # Merge back
        actor_critic = nnx.merge(ckpt["module_graphdef"], ckpt["module_state"])
        optimizers = nnx.merge(ckpt["optimizers_graphdef"], ckpt["optimizers_state"])

        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # Load states from dataset
    print("Loading states from dataset...")
    states, areas, actions = load_states_from_dataset(dataset)

    print(f"Loaded {len(states)} states from dataset")
    print(f"  Normalized area range: {areas.min():.3f} - {areas.max():.3f}")
    print(f"  Action distribution: Red={np.sum(actions==0)}, White={np.sum(actions==1)}, Blue={np.sum(actions==2)}")

    # Get normalization parameters
    print("Getting normalization parameters...")
    area_min, area_max = get_area_normalization_params()

    # Compute Q-values using the Q network for each action
    print("Computing Q-values for all actions at each state...")
    # Q network takes (state, action) pairs where action is an integer index
    num_actions = args.action_dim
    q_values_by_action = []

    for action_idx in range(num_actions):
        # Create action array with all states getting this action
        action_indices = jnp.full((len(states),), action_idx, dtype=jnp.int32)

        # Compute Q-values for this action (returns q_pi, q1, q2)
        q_pi, q1, q2 = actor_critic.q(states, action_indices)  # type: ignore[attr-defined]

        # Use q_pi (minimum of q1 and q2)
        q_values_by_action.append(np.array(q_pi))

    # Stack Q-values: shape (num_actions, num_states)
    q_values_by_action = np.stack(q_values_by_action, axis=0)

    print(f"Q-value range across all actions: {q_values_by_action.min():.3f} - {q_values_by_action.max():.3f}")
    for action_idx in range(num_actions):
        print(f"  Action {action_idx}: {q_values_by_action[action_idx].min():.3f} - {q_values_by_action[action_idx].max():.3f}")

    # Create output directory if needed
    output_path = checkpoint_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot results
    print("Creating plot...")
    plot_value_vs_area(
        areas,
        q_values_by_action,
        output_path,
        area_min,
        area_max,
        bin_size=args.bin_size,
    )

    print("\nDone!")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
