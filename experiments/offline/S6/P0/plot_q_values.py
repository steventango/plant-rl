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

from representations.RichTileCoder import RichTileCoder, RichTileCoderConfig

experiment_paths = Path(__file__).parent.glob("*.json")
for experiment_path in experiment_paths:
    checkpoint_path = Path("checkpoints/results") / experiment_path.relative_to(Path("experiments").absolute()).with_suffix("") / "0/chk.pkl.xz"
    with open(experiment_path, "r") as f:
        config = json.load(f)
        representation_config = config["metaParameters"]["representation"]
    with lzma.open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    weights = checkpoint["a"].w

    tc_config = RichTileCoderConfig(
        tiles=representation_config["tiles"], tilings=representation_config["tilings"], dims=2, strategy=representation_config["strategy"]
    )

    tile_coder = RichTileCoder(tc_config)

    daytime_observation_space = np.linspace(0, 1, 12 * 6, endpoint=False)
    area_observation_space = np.linspace(0, 1, 100)

    num_actions = weights.shape[0]
    Q = np.full((len(daytime_observation_space), len(area_observation_space), num_actions), -10000.0)

    for i, area in enumerate(area_observation_space):
        for j, time in enumerate(daytime_observation_space):
            for action in range(num_actions):
                indices = tile_coder.get_indices(np.array([time, area]))
                Q[j, i, action] = weights[action, indices].sum()

    # Calculate the difference between Q[s, 1] and Q[s, 0]
    if num_actions >= 2:
        Q_diff = Q[:, :, 1] - Q[:, :, 0]
    else:
        print("Not enough actions to compute difference. Plotting Q[s,0] instead.")
        Q_diff = Q[:, :, 0]

    # plot Q value differences

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
    plt.show()
