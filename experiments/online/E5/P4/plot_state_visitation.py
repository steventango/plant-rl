# %%

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from representations.RichTileCoder import RichTileCoder, RichTileCoderConfig
from representations.tile3 import IHT

# %%
df = pd.read_csv("data.csv")
# %%


def to_numpy(string_list):
    """Converts a string representation of a list of numbers to a NumPy array."""
    if not isinstance(string_list, str):
        return string_list
    try:
        # Remove the brackets and split by space
        numbers_str = string_list.strip("[]").split()
        # Convert the strings to floats and create a NumPy array
        return np.array([float(num) for num in numbers_str])
    except AttributeError:
        return np.nan  # Or handle non-string elements as needed


df["state"] = df["state"].apply(to_numpy)

states = np.stack(df["state"].to_numpy())
actions = df["action"].to_numpy()


states.shape, actions.shape
# %%

# %%
# plot weights

# (4, 800)
# %%

config = RichTileCoderConfig(
    tiles=4,
    tilings=32,
    dims=2,
    wrap_time=True,
)


tile_coder = RichTileCoder(config)
tile_coder.maxSize = 80000
tile_coder.iht = IHT(tile_coder.maxSize)
times = np.linspace(0, 1, 24 * 12, endpoint=True)

# %%
areas = np.linspace(-1, 1, 100)


inverse_mapping = defaultdict(set)
for i, area in enumerate(areas):
    for j, time in enumerate(times):
        indices = tile_coder.get_indices(np.array([time, area]))
        for index in indices:
            inverse_mapping[index].add((i, j))

# %%
num_actions = np.unique(actions).shape[0]
N = np.zeros((len(times), len(areas), num_actions))

for state, action in zip(states, actions, strict=False):
    indices = tile_coder.get_indices(state)
    for index in indices:
        if index not in inverse_mapping:
            print(f"Index {index} not found in inverse mapping.")
            continue
        for area_index, time_index in inverse_mapping[index]:
            N[time_index, area_index, action] += 1


# %%
N /= 32
# %%
# plot Q values

fig, axs = plt.subplots(num_actions, 1, figsize=(6, 2 * num_actions), sharey=True)
action_labels = ["moonlight", "dim", "normal", "extrabright"]

# Find the min and max values for the color scale
vmin = N.min()
vmax = N.max()

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
        N[:, :, action].T,  # Transpose Q to swap axes
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
    ax.set_xticks(np.arange(0, len(times) + 1, hour_interval))  # Set ticks every hour
    ax.set_xticklabels(
        [f"{int(t * 24)}" for t in times[::hour_interval]] + [24]
    )  # Format as hours
    ax.set_yticks(np.arange(0, len(areas) + 1, 20))  # Set y ticks every 10 areas
    ax.set_yticklabels(
        [f"{area:.1f}" for area in areas[::20]] + [f"{1:.1f}"]
    )  # Format as float
    ax.invert_yaxis()  # Invert the y-axis

fig.suptitle("ESARSA(λ) N(s, a)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
plt.show()
# %%
# also plot ESARSA(λ) N(s)

Ns = np.sum(N, axis=2)
fig, ax = plt.subplots(figsize=(6, 2))
sns.heatmap(
    Ns.T,  # Transpose Q to swap axes
    ax=ax,
    cmap=custom_cmap,  # Use the modified viridis colormap
    cbar=True,  # Add colorbar only to the last subplot
    vmin=vmin,
    vmax=vmax,
)
ax.set_title("ESARSA(λ) N(s)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Area")  # Area is now on the y-axis
ax.set_xticks(np.arange(0, len(times), hour_interval))  # Set ticks every hour
ax.set_xticklabels(
    [f"{int(t * 24)}" for t in times[::hour_interval]]
)  # Format as hours
ax.set_yticks(np.arange(0, len(areas), 20))  # Set y ticks every 10 areas
ax.set_yticklabels([f"{area:.1f}" for area in areas[::20]])  # Format as float
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
ax.invert_yaxis()  # Invert the y-axis
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
plt.show()


# %%
# plot a snake (gradient line) of state visitation from the first to the last state

fig, ax = plt.subplots(figsize=(6, 6))
previous_closest_time = None
previous_closest_area = None
for i, state in enumerate(states):
    closest_time = np.argmin(np.abs(times - state[0]))
    closest_area = np.argmin(np.abs(areas - state[1]))
    if previous_closest_time is not None and previous_closest_area is not None:
        # Draw a line segment between the previous and current points
        ax.plot(
            [previous_closest_time, closest_time],
            [previous_closest_area, closest_area],
            color=plt.cm.viridis(i / len(states)),
            alpha=0.5,
            linewidth=1,
        )
    # Update the previous points
    previous_closest_time = closest_time
    previous_closest_area = closest_area
ax.set_title("ESARSA(λ) N(s)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Area")  # Area is now on the y-axis
ax.set_xticks(np.arange(0, len(times), hour_interval))  # Set ticks every hour
ax.set_xticklabels(
    [f"{int(t * 24)}" for t in times[::hour_interval]]
)  # Format as hours
ax.set_yticks(np.arange(0, len(areas), 20))  # Set y ticks every 10 areas
ax.set_yticklabels([f"{area:.1f}" for area in areas[::20]])  # Format as float
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
# ax.invert_yaxis()  # Invert the y-axis
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
plt.show()


# %%
