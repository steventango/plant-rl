import os
import sys

sys.path.append(os.getcwd() + '/src')

import json
import lzma
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

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

    daytime_observation_space = np.linspace(0, 1, 24 * 12, endpoint=False)
    area_observation_space = np.linspace(0, 1, 100)

    num_actions = weights.shape[0]
    Q = np.full((len(daytime_observation_space), len(area_observation_space), num_actions), -10000.0)

    for i, area in enumerate(area_observation_space):
        for j, time in enumerate(daytime_observation_space):
            for action in range(num_actions):
                indices = tile_coder.get_indices(np.array([time, area]))
                Q[j, i, action] = weights[action, indices].sum()

    # %%
    # plot Q values

    fig, axs = plt.subplots(num_actions, 1, figsize=(6, 2 * num_actions), sharey=True)
    action_labels = ["moonlight", "dim", "normal", "extrabright"]

    # Find the min and max values for the color scale
    vmin = Q.min()
    vmax = Q.max()

    # Calculate tick interval for every hour
    hour_interval = 12

    # Create a modified viridis colormap with gray for zero
    viridis = plt.cm.get_cmap("viridis")(np.linspace(0, 1, 256))
    # from vmax and vmin find the position of zero in the colormap
    zero_position = int((0 - vmin) / (vmax - vmin) * 255)

    viridis[zero_position, :3] = [0.5, 0.5, 0.5]  # Set RGB values for gray
    custom_cmap = ListedColormap(viridis)

    # Create the heatmaps
    for action, ax in zip(range(num_actions), axs):
        sns.heatmap(
            Q[:, :, action].T,  # Transpose Q to swap axes
            ax=ax,
            cmap=custom_cmap,  # Use the modified viridis colormap
            cbar=(action == num_actions - 1),  # Add colorbar only to the last subplot
            vmin=vmin,
            vmax=vmax,
            cbar_ax=None if action < num_actions - 1 else fig.add_axes([0.92, 0.15, 0.02, 0.7]),
        )
        ax.set_title(f"Action {action} ({action_labels[action]})")
        ax.set_xlabel("Time (h)" if action == num_actions - 1 else "")
        ax.set_ylabel("Area")  # Area is now on the y-axis
        ax.set_xticks(np.arange(0, len(daytime_observation_space), hour_interval))  # Set ticks every hour
        ax.set_xticklabels([f"{int(t * 24)}" for t in daytime_observation_space[::hour_interval]])  # Format as hours
        ax.set_yticks(np.arange(0, len(area_observation_space), 10))  # Set y ticks every 10 areas
        ax.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])  # Format as float
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
        ax.invert_yaxis()  # Invert the y-axis

    fig.suptitle("ESARSA(λ) Q-values", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.show()
