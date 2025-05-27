import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def plot(ax, data, label=None):
    mean, ste, runs = data
    base, = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)


def get_Q(weights, tile_coder, daytime_observation_space, area_observation_space, num_actions):
    Q = np.full((len(daytime_observation_space), len(area_observation_space), num_actions), -10000.0)

    for i, area in enumerate(area_observation_space):
        for j, time in enumerate(daytime_observation_space):
            for action in range(num_actions):
                indices = tile_coder.get_indices(np.array([time, area]))
                Q[j, i, action] = weights[action, indices].sum()
    return Q


def plot_q_diff(daytime_observation_space, area_observation_space, Q_diff):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Two subplots: one for Q-diff, one for policy

    # Plot Q-difference (similar to before)
    ax_q_diff = axs[0]
    eps = 1e-5
    vmin = min(Q_diff.min(), -eps)
    vmax = max(Q_diff.max(), eps)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    sns.heatmap(
        Q_diff.T,
        ax=ax_q_diff,
        cmap="bwr",
        cbar=True,
        norm=norm,
        cbar_ax=fig.add_axes([0.47, 0.15, 0.02, 0.7]), # Adjusted cbar position
    )
    ax_q_diff.set_title(f"Q(s, a=1) - Q(s, a=0)")
    ax_q_diff.set_xlabel("Time")
    hour_interval = 6
    ax_q_diff.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))
    ax_q_diff.set_xticklabels(
        [f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21]
    )
    ax_q_diff.set_ylabel("Area")
    ax_q_diff.set_yticks(np.arange(0, len(area_observation_space), 10))
    ax_q_diff.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])
    ax_q_diff.set_yticklabels(ax_q_diff.get_yticklabels(), rotation=0)
    ax_q_diff.invert_yaxis()

    # Plot Policy
    ax_policy = axs[1]
    policy = (Q_diff > 0).astype(int)  # 1 if Q(s,1) > Q(s,0), else 0

    # Create a discrete colormap for the policy: 0 (blue), 1 (red)
    cmap_policy = mcolors.ListedColormap(['blue', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm_policy = mcolors.BoundaryNorm(bounds, cmap_policy.N)


    sns.heatmap(
        policy.T,
        ax=ax_policy,
        cmap=cmap_policy,
        norm=norm_policy,
        cbar=True,
        cbar_kws={"ticks": [0, 1]}, # Ticks for the policy colorbar
        cbar_ax=fig.add_axes([0.92, 0.15, 0.02, 0.7]), # Adjusted cbar position
    )
    ax_policy.set_title("Policy (Preferred Action)")
    ax_policy.set_xlabel("Time")
    ax_policy.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))
    ax_policy.set_xticklabels(
        [f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21]
    )
    ax_policy.set_ylabel("Area")
    ax_policy.set_yticks(np.arange(0, len(area_observation_space), 10))
    ax_policy.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])
    ax_policy.set_yticklabels(ax_policy.get_yticklabels(), rotation=0)
    ax_policy.invert_yaxis()

    # Set labels for policy colorbar
    cbar = ax_policy.collections[0].colorbar
    cbar.set_ticklabels(['Action 0', 'Action 1'])


    fig.suptitle("ESARSA(λ) Q-value Difference and Policy", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbars


def plot_q(daytime_observation_space, area_observation_space, Q):
    num_actions = Q.shape[2]
    fig, axs = plt.subplots(1, num_actions, figsize=(6 * num_actions, 4))
    if num_actions == 1:
        axs = [axs]  # Ensure axs is iterable even for a single subplot

    # Find the global min and max Q values for a consistent color scale
    eps = 1e-5  # Small epsilon to avoid division by zero
    # vmin = min(Q.min(), -eps)  # Ensure vmin is not less than -eps
    # vmax = max(Q.max(), eps)  # Ensure vmax is not greater than eps

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
            # vmin=vmin,
            # vmax=vmax,
        )
        ax.set_title(f"Q(s, {i})")

        ax.set_xlabel("Time")
        hour_interval = 6
        ax.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))  # Set ticks every hour
        ax.set_xticklabels(
            [f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21]
        )  # Format as hours

        ax.set_ylabel("Area")  # Area is now on the y-axis
        ax.set_yticks(np.arange(0, len(area_observation_space), 10))  # Set y ticks every 10 areas
        ax.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])  # Format as float
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
        ax.invert_yaxis()  # Invert the y-axis

    fig.suptitle("ESARSA(λ) Q-values", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
