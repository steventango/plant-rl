import os
import sys

sys.path.append(os.getcwd() + '/src')

import json
import lzma
import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_q(daytime_observation_space, area_observation_space, Q):
    num_actions = Q.shape[2]
    fig, axs = plt.subplots(1, num_actions, figsize=(6 * num_actions, 4))
    if num_actions == 1:
        axs = [axs]  # Ensure axs is iterable even for a single subplot

    # Find the global min and max Q values for a consistent color scale
    vmin = Q.min()
    vmax = Q.max()

    # Create a single colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, ax in enumerate(axs):
        # Create the heatmap
        sns.heatmap(
            Q[:, :, i].T,
            ax=ax,
            cmap="viridis",
            cbar=i == num_actions - 1,  # Only add cbar to the last plot
            cbar_ax=cbar_ax if i == num_actions - 1 else None,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"Q(s, {i})")

        ax.set_xlabel("Time")
        hour_interval = 6
        ax.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))  # Set ticks every hour
        ax.set_xticklabels([f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21])  # Format as hours

        ax.set_ylabel("Area")  # Area is now on the y-axis
        ax.set_yticks(np.arange(0, len(area_observation_space), 10))  # Set y ticks every 10 areas
        ax.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])  # Format as float
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
        ax.invert_yaxis()  # Invert the y-axis

    fig.suptitle("ESARSA(λ) Q-values", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.savefig("Q_values.png", dpi=300, bbox_inches='tight')


def plot_q_diff(daytime_observation_space, area_observation_space, Q_diff):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4)) # Single plot

    # Find the min and max values for the color scale from Q_diff
    vmin = Q_diff.min()
    vmax = Q_diff.max()

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

    # Create the heatmap for the difference
    sns.heatmap(
        Q_diff.T,  # Transpose Q_diff to swap axes
        ax=ax,
        cmap="bwr",
        cbar=True,
        norm=norm,
        cbar_ax=fig.add_axes([0.92, 0.15, 0.02, 0.7]),
    )
    ax.set_title(f"Q(s, action=1) - Q(s, action=0)")

    ax.set_xlabel("Time")
    hour_interval = 6
    ax.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))  # Set ticks every hour
    ax.set_xticklabels([f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21])  # Format as hours

    ax.set_ylabel("Area")  # Area is now on the y-axis
    ax.set_yticks(np.arange(0, len(area_observation_space), 10))  # Set y ticks every 10 areas
    ax.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])  # Format as float
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
    ax.invert_yaxis()  # Invert the y-axis

    fig.suptitle("ESARSA(λ) Q-value Difference", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.savefig("Q_value_difference.png", dpi=300, bbox_inches='tight')



experiment_paths = Path(__file__).parent.glob("*.json")

for experiment_path in experiment_paths:
    checkpoint_path = Path("checkpoints/results") / experiment_path.relative_to(Path("experiments")).with_suffix("") / "0/chk.pkl.xz"
    if not checkpoint_path.exists():
        print(f"Checkpoint file {checkpoint_path} does not exist. Skipping {experiment_path}.")
        continue
    with open(experiment_path, "r") as f:
        config = json.load(f)
        representation_config = config["metaParameters"]["representation"]
    with lzma.open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    weights = checkpoint["a"].w

    tile_coder = checkpoint["a"].tile_coder

    daytime_observation_space = np.linspace(0, 1, 12 * 6, endpoint=True)
    area_observation_space = np.linspace(0, 1, 100, endpoint=True)

    num_actions = weights.shape[0]
    Q = np.full((len(daytime_observation_space), len(area_observation_space), num_actions), -10000.0)

    for i, area in enumerate(area_observation_space):
        for j, time in enumerate(daytime_observation_space):
            for action in range(num_actions):
                indices = tile_coder.get_indices(np.array([time, area]))
                Q[j, i, action] = weights[action, indices].sum()


    plot_q(daytime_observation_space, area_observation_space, Q)

    # Calculate the difference between Q[s, 1] and Q[s, 0]
    if num_actions >= 2:
        Q_diff = Q[:, :, 1] - Q[:, :, 0]
    else:
        print("Not enough actions to compute difference. Plotting Q[s,0] instead.")
        Q_diff = Q[:, :, 0]

    # plot Q value differences
    plot_q_diff(daytime_observation_space, area_observation_space, Q_diff)
