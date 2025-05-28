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
    eps = 1e-5  # Small epsilon to avoid division by zero
    vmin = min(Q_diff.min(), -eps)  # Ensure vmin is not less than -eps
    vmax = max(Q_diff.max(), eps)   # Ensure vmax is not greater than eps
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    sns.heatmap(
        Q_diff.T,
        ax=ax_q_diff,
        cmap="bwr",
        cbar=True,
        norm=norm
    )

    ax_q_diff.set_title(f"Q(s, a=1) - Q(s, a=0)")
    ax_q_diff.set_xlabel("s[0]")
    hour_interval = 6
    ax_q_diff.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))
    ax_q_diff.set_xticklabels(
        [f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21]
    )
    ax_q_diff.set_ylabel("s[1]")
    ax_q_diff.set_yticks(np.arange(0, len(area_observation_space), 10))
    ax_q_diff.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])
    ax_q_diff.set_yticklabels(ax_q_diff.get_yticklabels(), rotation=0)
    ax_q_diff.invert_yaxis()

    # Plot Policy
    ax_policy = axs[1]
    policy_cutoff = 1e-1
    policy = (Q_diff > policy_cutoff).astype(int) - (Q_diff < -policy_cutoff).astype(int) 

    # Create a discrete colormap for the policy: -1 (blue), 0 (gray), 1 (red),
    cmap_policy = mcolors.ListedColormap(['blue', 'gray', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_policy = mcolors.BoundaryNorm(bounds, cmap_policy.N)


    sns.heatmap(
        policy.T,
        ax=ax_policy,
        cmap=cmap_policy,
        norm=norm_policy,
        cbar=True,
        cbar_kws={"ticks": [-1, 0, 1]}, # Ticks for the policy colorbar
    )
    ax_policy.set_title(rf"Policy (Preferred Action, $\Delta$Q={policy_cutoff})")
    ax_policy.set_xlabel("s[0]")
    ax_policy.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))
    ax_policy.set_xticklabels(
        [f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21]
    )
    ax_policy.set_ylabel("s[1]")
    ax_policy.set_yticks(np.arange(0, len(area_observation_space), 10))
    ax_policy.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])
    ax_policy.set_yticklabels(ax_policy.get_yticklabels(), rotation=0)
    ax_policy.invert_yaxis()

    # Set labels for policy colorbar
    cbar = ax_policy.collections[0].colorbar
    cbar.set_ticklabels(['action 0', 'hard to tell', 'action 1'])

    fig.suptitle("ESARSA Q-value Difference and Policy", fontsize=16)


def plot_q(daytime_observation_space, area_observation_space, Q):
    num_actions = Q.shape[2]
    fig, axs = plt.subplots(1, num_actions, figsize=(6 * num_actions, 4))
    if num_actions == 1:
        axs = [axs]  # Ensure axs is iterable even for a single subplot

    # Find the global min and max Q values for a consistent color scale
    vmin = Q.min()
    vmax = Q.max()

    for i, ax in enumerate(axs):
        # Create the heatmap
        sns.heatmap(
            Q[:, :, i].T,
            ax=ax,
            cmap="viridis",
            cbar = True,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Q(s, {i})")

        ax.set_xlabel("s[0]")
        hour_interval = 6
        ax.set_xticks(np.arange(0, len(daytime_observation_space) + 1, hour_interval))  # Set ticks every hour
        ax.set_xticklabels(
            [f"{int(t * 12) + 9}" for t in daytime_observation_space[::hour_interval]] + [21]
        )  # Format as hours

        ax.set_ylabel("s[1]")  # Area is now on the y-axis
        ax.set_yticks(np.arange(0, len(area_observation_space), 10))  # Set y ticks every 10 areas
        ax.set_yticklabels([f"{area:.1f}" for area in area_observation_space[::10]])  # Format as float
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y ticks
        ax.invert_yaxis()  # Invert the y-axis

    fig.suptitle("ESARSA Q-values", fontsize=16)
