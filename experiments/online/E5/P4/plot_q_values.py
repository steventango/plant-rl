# %%

import lzma
import pickle

# checkpoints/results/online/E5/P4/ESARSA/0/chk.pkl.xz

with lzma.open("checkpoints/results/online/E5/P4/ESARSA/0/chk.pkl.xz", "rb") as f:
    data = pickle.load(f)

print(data)
# %%
weights = data["a"].w

# %%

# plot weights

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

# (4, 800)


# %%


# count of nans
def count_nans(weights):
    return np.isnan(weights).sum()


nan_count = count_nans(data["a"].w)
print(f"Number of NaNs in weights: {nan_count}")


# count of infs
def count_infs(weights):
    return np.isinf(weights).sum()


inf_count = count_infs(data["a"].w)
print(f"Number of Infs in weights: {inf_count}")


# count of zeros
def count_zeros(weights):
    return np.count_nonzero(weights == 0)


zero_count = count_zeros(data["a"].w)
print(f"Number of Zeros in weights: {zero_count}")


# %%
# map indices back to original features
from src.representations.RichTileCoder import RichTileCoder, RichTileCoderConfig

# "representation": {
#       "tiles": 4,
#       "tilings": 32,
#       "which_tc": "RichTileCoder",
#       "wrap_time": true
#     }

config = RichTileCoderConfig(
    tiles=4,
    tilings=32,
    dims=2,
    wrap_time=True,
)

tile_coder = RichTileCoder(config)

times = np.linspace(0, 1, 24 * 12, endpoint=False)

# %%
areas = np.linspace(0, 1, 100)

weights = data["a"].w
num_actions = weights.shape[0]
Q = np.full((len(times), len(areas), num_actions), -10000.0)

for i, area in enumerate(areas):
    for j, time in enumerate(times):
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
viridis = plt.cm.viridis(np.linspace(0, 1, 256))
# from vmax and vmin find the position of zero in the colormap
zero_position = int((0 - vmin) / (vmax - vmin) * 255)

viridis[zero_position, :3] = [0.5, 0.5, 0.5]  # Set RGB values for gray
custom_cmap = ListedColormap(viridis)

# Create the heatmaps
for action, ax in zip(range(num_actions), axs, strict=False):
    sns.heatmap(
        Q[:, :, action].T,  # Transpose Q to swap axes
        ax=ax,
        cmap=custom_cmap,  # Use the modified viridis colormap
        cbar=(action == num_actions - 1),  # Add colorbar only to the last subplot
        vmin=vmin,
        vmax=vmax,
        cbar_ax=None
        if action < num_actions - 1
        else fig.add_axes([0.92, 0.15, 0.02, 0.7]),
    )
    ax.set_title(f"Action {action} ({action_labels[action]})")
    ax.set_xlabel("Time (h)" if action == num_actions - 1 else "")
    ax.set_ylabel("Area")  # Area is now on the y-axis
    ax.set_xticks(np.arange(0, len(times), hour_interval))  # Set ticks every hour
    ax.set_xticklabels(
        [f"{int(t * 24)}" for t in times[::hour_interval]]
    )  # Format as hours
    ax.set_yticks(np.arange(0, len(areas), 10))  # Set y ticks every 10 areas
    ax.set_yticklabels([f"{area:.1f}" for area in areas[::10]])  # Format as float
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
    ax.invert_yaxis()  # Invert the y-axis

fig.suptitle("ESARSA(λ) Q-values", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
plt.show()

# %%
