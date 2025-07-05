# %%
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from utils.calibration import load_and_clean_data


def plot_spectral_data(
    df: pd.DataFrame, zone: int, ax: Axes, colors: np.ndarray
) -> None:
    """Plots spectral data for a given zone."""
    sorted_action_values = df.columns[1:].tolist()

    # Add traces to the subplot
    for i, action in enumerate(sorted_action_values):
        ax.plot(
            df["Wavelength"],
            df[action],
            label=f"Action {action}",
            color=colors[i],
        )
    ax.set_title(f"Zone {zone}")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Intensity")
    ax.grid(True)


# Load the calibration data
calibration_file = (
    "/workspaces/plant-rl/scripts/calibration/Plant Chamber Full Calibration.xlsx"
)

# Create subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle("Far Red Spectral Data for All Zones")

num_actions = 13
colors = cm.get_cmap("viridis")(
    [i / num_actions for i in range(num_actions)]
)  # Use a colormap

for zone in range(1, 13):
    spectral_file = os.path.join(
        os.path.dirname(calibration_file), f"RL_FarRedIntensityz{zone}.txt"
    )
    df, integrals = load_and_clean_data(spectral_file)

    row = (zone - 1) // 3
    col = (zone - 1) % 3
    ax = axes[row, col]
    plot_spectral_data(df, zone, ax, colors)

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

plt.show()

# %%
