import os

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)


def plot(ax, data, label=None):
    mean, ste, runs = data
    (base,) = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(
        range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4
    )


def get_Q(
    weights, tile_coder, daytime_observation_space, area_observation_space, num_actions
):
    Q = np.full(
        (len(daytime_observation_space), len(area_observation_space), num_actions),
        -10000.0,
    )

    for i, area in enumerate(area_observation_space):
        for j, time in enumerate(daytime_observation_space):
            for action in range(num_actions):
                indices = tile_coder.get_indices(np.array([time, area]))
                Q[j, i, action] = weights[action, indices].sum()
    return Q


def plot_q_diff(daytime_observation_space, area_observation_space, Q_diff):
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5)
    )  # Two subplots: one for Q-diff, one for policy

    # Plot Q-difference (similar to before)
    ax_q_diff = axs[0]
    eps = 1e-5  # Small epsilon to avoid division by zero
    vmin = min(Q_diff.min(), -eps)  # Ensure vmin is not less than -eps
    vmax = max(Q_diff.max(), eps)  # Ensure vmax is not greater than eps
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    sns.heatmap(Q_diff.T, ax=ax_q_diff, cmap="bwr", cbar=True, norm=norm)
    if ax_q_diff.images:
        ax_q_diff.set_aspect("auto")

    ax_q_diff.set_title("Q(s, a=1) - Q(s, a=0)")
    ax_q_diff.set_xlabel("s[0]")
    num_daytime_bins = len(daytime_observation_space)
    xtick_indices = np.linspace(0, num_daytime_bins - 1, 7, dtype=int)
    xtick_labels_values = np.linspace(0, 1, 7)
    ax_q_diff.set_xticks(xtick_indices)
    ax_q_diff.set_xticklabels([f"{int(t * 11) + 9}" for t in xtick_labels_values])

    ax_q_diff.set_ylabel("s[1]")
    num_area_bins = len(area_observation_space)
    ytick_indices = np.linspace(0, num_area_bins - 1, 6, dtype=int)
    ytick_labels_values = np.linspace(0, 1, 6)
    ax_q_diff.set_yticks(ytick_indices)
    ax_q_diff.set_yticklabels(
        [f"{area:.1f}" for area in ytick_labels_values], rotation=0
    )
    ax_q_diff.invert_yaxis()

    # Plot Policy
    ax_policy = axs[1]
    policy_cutoff = 1e-10
    policy = (Q_diff > policy_cutoff).astype(int) - (Q_diff < -policy_cutoff).astype(
        int
    )

    # Create a discrete colormap for the policy: -1 (blue), 0 (gray), 1 (red),
    cmap_policy = mcolors.ListedColormap(["blue", "gray", "red"])
    norm_policy = mcolors.BoundaryNorm(
        boundaries=[-1.5, -0.5, 0.5, 1.5],
        ncolors=3,
    )  # Boundaries for the policy colormap

    sns.heatmap(
        policy.T,
        ax=ax_policy,
        cmap=cmap_policy,
        norm=norm_policy,
        cbar=True,
        cbar_kws={"ticks": [-1, 0, 1]},  # Ticks for the policy colorbar
    )
    if ax_policy.images:
        ax_policy.set_aspect("auto")

    ax_policy.set_title("Policy")
    ax_policy.set_xlabel("s[0]")
    ax_policy.set_xticks(xtick_indices)
    ax_policy.set_xticklabels([f"{int(t * 11) + 9}" for t in xtick_labels_values])
    ax_policy.set_ylabel("s[1]")
    ax_policy.set_yticks(ytick_indices)
    ax_policy.set_yticklabels(
        [f"{area:.1f}" for area in ytick_labels_values], rotation=0
    )
    ax_policy.invert_yaxis()

    # Set labels for policy colorbar
    if ax_policy.collections:
        cbar = ax_policy.collections[0].colorbar
        cbar.set_ticklabels(["A=0", "N/A", "A=1"])
    fig.tight_layout()
    fig.suptitle("Q-value Difference and Policy", fontsize=16)
    return fig, axs


def plot_q(daytime_observation_space, area_observation_space, Q):
    num_actions = Q.shape[2]
    fig, axs = plt.subplots(1, num_actions, figsize=(6 * num_actions, 5))
    if num_actions == 1:
        axs = [axs]  # Ensure axs is iterable even for a single subplot

    # Find the global min and max Q values for a consistent color scale
    vmin = Q.min()
    vmax = Q.max()

    for i, ax in enumerate(axs):  # type: ignore
        # Create the heatmap
        sns.heatmap(
            Q[:, :, i].T,
            ax=ax,
            cmap="viridis",
            cbar=True,
            vmin=vmin,
            vmax=vmax,
        )
        if ax.images:
            ax.set_aspect("auto")

        ax.set_title(f"Q(s, {i})")

        ax.set_xlabel("s[0]")
        num_daytime_bins = len(daytime_observation_space)
        xtick_indices = np.linspace(0, num_daytime_bins - 1, 7, dtype=int)
        xtick_labels_values = np.linspace(0, 1, 7)
        ax.set_xticks(xtick_indices)
        ax.set_xticklabels([f"{int(t * 11) + 9}" for t in xtick_labels_values])

        ax.set_ylabel("s[1]")
        num_area_bins = len(area_observation_space)
        ytick_indices = np.linspace(0, num_area_bins - 1, 6, dtype=int)
        ytick_labels_values = np.linspace(0, 1, 6)
        ax.set_yticks(ytick_indices)
        ax.set_yticklabels([f"{area:.1f}" for area in ytick_labels_values], rotation=0)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.suptitle("Q-values", fontsize=16)
    return fig, axs


def _prepare_trajectory_df(df):
    """Prepares a DataFrame for trajectory plotting."""
    # Filter out rows where observation is invalid
    plot_df = df[
        df["observation"].notna()
        & df["observation"].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) >= 2
        )
    ].copy()

    # Identify trajectory boundaries
    plot_df["trajectory_id"] = plot_df["terminal"].isnull().cumsum()
    if "trajectory_name" in plot_df.columns:
        plot_df["trajectory_name"] = (
            plot_df["trajectory_name"].fillna(method="ffill").fillna(method="bfill")
        )
    else:
        plot_df["trajectory_name"] = None

    # Shift observations to create trajectory segments
    plot_df["next_observation"] = plot_df.groupby("trajectory_id")[
        "observation"
    ].transform(lambda x: x.shift(-1))
    plot_df = plot_df[
        plot_df["next_observation"].notna()
        & plot_df["next_observation"].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) >= 2
        )
    ].copy()

    if not plot_df.empty:
        # Extract state components
        plot_df["daytime"] = plot_df["observation"].apply(lambda x: x[0])
        plot_df["area"] = plot_df["observation"].apply(lambda x: x[1])
        plot_df["next_daytime"] = plot_df["next_observation"].apply(lambda x: x[0])
        plot_df["next_area"] = plot_df["next_observation"].apply(lambda x: x[1])

    return plot_df


def _plot_single_trajectory_on_ax(ax, traj_df, scale=1):
    """Plots a single trajectory on a given matplotlib axes."""
    num_segments = len(traj_df)
    actions = (
        traj_df["action"].values
        if "action" in traj_df.columns
        else np.zeros(num_segments)
    )
    time_norm = np.linspace(0, 1, num_segments)
    reds = plt.get_cmap("Reds")
    blues = plt.get_cmap("Blues")

    for j in range(num_segments):
        action = actions[j] if not pd.isna(actions[j]) else 0
        if action == 1:
            color = reds(time_norm[j] * 0.7 + 0.3)
        else:
            color = blues(time_norm[j] * 0.7 + 0.3)

        daytime_scaled = traj_df["daytime"].iloc[j] * scale
        area_scaled = traj_df["area"].iloc[j] * scale
        next_daytime_scaled = traj_df["next_daytime"].iloc[j] * scale
        next_area_scaled = traj_df["next_area"].iloc[j] * scale

        ax.quiver(
            daytime_scaled,
            area_scaled,
            next_daytime_scaled - daytime_scaled,
            next_area_scaled - area_scaled,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.7,
        )


def plot_q_values_and_diff(
    logger, agent, q_plots_dir, step, df
):  # Renamed original plot function
    if (
        hasattr(agent, "w")
        and hasattr(agent, "tile_coder")
        and agent.w is not None
        and agent.tile_coder is not None
    ):
        try:
            weights = agent.w
            tile_coder = agent.tile_coder

            # Define observation space for plotting (as in plot_q_values.py)
            daytime_observation_space = np.linspace(0, 1, 11 * 6, endpoint=False)
            area_observation_space = np.linspace(0, 1, 11 * 6, endpoint=True)

            num_actions = weights.shape[0]
            Q_vals = get_Q(
                weights,
                tile_coder,
                daytime_observation_space,
                area_observation_space,
                num_actions,
            )

            # Plot and save Q-values
            q_plot_filename = q_plots_dir / f"q_values_step_{step:06d}.jpg"
            fig_q, _ = plot_q(
                daytime_observation_space, area_observation_space, Q_vals
            )  # Call the plot function
            fig_q.savefig(q_plot_filename)  # Save the figure
            plt.close(fig_q)  # Close the figure

            # Plot and save Q-value differences
            if num_actions >= 2:
                Q_diff = Q_vals[:, :, 1] - Q_vals[:, :, 0]
            else:
                logger.info(
                    f"Step {step}: Not enough actions ({num_actions}) to compute Q-difference. Plotting Q[s,0] instead."
                )
                Q_diff = Q_vals[:, :, 0]

            q_diff_plot_filename = q_plots_dir / f"q_diff_step_{step:06d}.jpg"
            fig_diff, axs_diff = plot_q_diff(
                daytime_observation_space, area_observation_space, Q_diff
            )  # Call the plot function

            # PLOT TRAJECTORIES ON TOP
            try:
                plot_df = _prepare_trajectory_df(df)
                if not plot_df.empty:
                    trajectory_ids = plot_df["trajectory_id"].unique()
                    ax = axs_diff[1]
                    for traj_id in trajectory_ids:
                        traj_df = plot_df[plot_df["trajectory_id"] == traj_id]
                        _plot_single_trajectory_on_ax(ax, traj_df, scale=11 * 6)
            except Exception as e:
                logger.error(
                    f"Step {step}: Error during trajectory overlay plotting: {e}",
                    exc_info=True,
                )

            fig_diff.savefig(q_diff_plot_filename)  # Save the figure
            plt.close(fig_diff)  # Close the figure
            logger.info(f"Step {step}: Saved Q-value plots to {q_plots_dir}")
        except Exception as e:
            logger.error(
                f"Step {step}: Error during Q-value plotting: {e}", exc_info=True
            )
    else:
        logger.warning(
            f"Step {step}: Agent does not have 'w' or 'tile_coder' attributes, or they are None. Skipping Q-value plotting."
        )


def plot_state_action_distribution(df, q_plots_dir, logger):
    try:
        # Filter out rows where action is None and ensure action is numeric
        plot_df = df[df["action"].notna()].copy()
        plot_df["action"] = pd.to_numeric(plot_df["action"], errors="coerce")
        plot_df.dropna(subset=["action"], inplace=True)

        if not plot_df.empty:
            # Extract state components
            plot_df["daytime"] = plot_df["observation"].apply(
                lambda x: x[0]
                if isinstance(x, (list, np.ndarray)) and len(x) > 0
                else np.nan
            )
            plot_df["area"] = plot_df["observation"].apply(
                lambda x: x[1]
                if isinstance(x, (list, np.ndarray)) and len(x) > 1
                else np.nan
            )
            plot_df.dropna(subset=["daytime", "area"], inplace=True)

            if not plot_df.empty:
                # Define bins consistent with Q-value plots
                num_daytime_bins = 11
                num_area_bins = 11

                daytime_bin_edges = np.linspace(
                    0, 1, num_daytime_bins + 1, endpoint=True
                )
                area_bin_edges = np.linspace(0, 1, num_area_bins + 1, endpoint=True)

                daytime_bin_labels = range(num_daytime_bins)
                area_bin_labels = range(num_area_bins)

                plot_df["daytime_bin"] = pd.cut(
                    plot_df["daytime"],
                    bins=daytime_bin_edges,
                    labels=daytime_bin_labels,
                    include_lowest=True,
                    right=True,
                )
                plot_df["area_bin"] = pd.cut(
                    plot_df["area"],
                    bins=area_bin_edges,
                    labels=area_bin_labels,
                    include_lowest=True,
                    right=True,
                )
                plot_df.dropna(subset=["daytime_bin", "area_bin"], inplace=True)

                if not plot_df.empty:
                    actions_to_plot = sorted(
                        [
                            action
                            for action in plot_df["action"].unique()
                            if action in [0.0, 1.0]
                        ]
                    )
                    if not actions_to_plot:
                        logger.info("No actions 0 or 1 found in the data to plot.")
                        return

                    num_subplots = len(actions_to_plot)
                    fig, axs = plt.subplots(
                        1, num_subplots, figsize=(6 * num_subplots, 5), squeeze=False
                    )
                    fig.suptitle(
                        "State-Action Distribution (Count per Bin)", fontsize=16
                    )

                    # Determine global max count for consistent color scaling if desired, or use individual scales
                    # For now, using individual scales by not setting vmax explicitly for sns.heatmap or setting it per subplot.
                    # global_max_count = 0
                    # for action_val in actions_to_plot:
                    #     action_df = plot_df[plot_df["action"] == action_val]
                    #     if not action_df.empty:
                    #         counts = action_df.groupby(["daytime_bin", "area_bin"]).size()
                    #         if not counts.empty:
                    #             global_max_count = max(global_max_count, counts.max())

                    for i, action_val in enumerate(actions_to_plot):
                        ax = axs[0, i]
                        action_df = plot_df[plot_df["action"] == action_val]

                        if not action_df.empty:
                            heatmap_data = (
                                action_df.groupby(["daytime_bin", "area_bin"])
                                .size()  # Get counts
                                .unstack(
                                    fill_value=0
                                )  # Fill non-observed bins with 0 count
                            )
                            # Ensure all bins are present
                            heatmap_data = heatmap_data.reindex(
                                index=daytime_bin_labels,
                                columns=area_bin_labels,
                                fill_value=0,
                            )

                            current_max_count = (
                                heatmap_data.max().max()
                                if not heatmap_data.empty
                                else 0
                            )

                            sns.heatmap(
                                heatmap_data.T,  # Transpose for daytime on x, area on y
                                ax=ax,
                                cmap="viridis",  # Or "Reds" for action 1, "Blues" for action 0 if preferred
                                vmin=0,
                                vmax=(
                                    current_max_count if current_max_count > 0 else 1
                                ),  # Avoid error if all counts are 0
                                cbar_kws={
                                    "label": f"Count of Action {int(action_val)}"
                                },
                                square=False,
                            )
                        else:
                            # If no data for this action, plot an empty heatmap or indicate no data
                            # For simplicity, seaborn will plot an empty grid if heatmap_data is all 0s or empty after reindex
                            empty_heatmap_data = pd.DataFrame(
                                0, index=daytime_bin_labels, columns=area_bin_labels
                            )
                            sns.heatmap(
                                empty_heatmap_data.T,
                                ax=ax,
                                cmap="viridis",
                                vmin=0,
                                vmax=1,
                                cbar_kws={
                                    "label": f"Count of Action {int(action_val)} (No Data)"
                                },
                                square=False,
                            )

                        ax.set_title(f"Distribution of Action {int(action_val)}")
                        ax.set_xlabel("s[0]")
                        ax.set_ylabel(
                            "s[1]" if i == 0 else ""
                        )  # Only label y-axis on the first plot

                        daytime_observation_space_for_labels = np.linspace(
                            0, 1, num_daytime_bins, endpoint=True
                        )
                        area_observation_space_for_labels = np.linspace(
                            0, 1, num_area_bins, endpoint=True
                        )
                        hour_interval = 6
                        xtick_indices = np.arange(0, num_daytime_bins, hour_interval)
                        ax.set_xticks(xtick_indices + 0.5)
                        xtick_labels_values = daytime_observation_space_for_labels[
                            ::hour_interval
                        ]
                        ax.set_xticklabels(
                            [f"{int(t * 11) + 9}" for t in xtick_labels_values]
                        )

                        area_tick_interval = 10
                        ytick_indices = np.arange(0, num_area_bins, area_tick_interval)
                        ax.set_yticks(ytick_indices + 0.5)
                        ytick_labels_values = area_observation_space_for_labels[
                            ::area_tick_interval
                        ]
                        ax.set_yticklabels(
                            [f"{area:.1f}" for area in ytick_labels_values], rotation=0
                        )

                        ax.invert_yaxis()

                    plt.tight_layout()  # Adjust layout to make space for suptitle
                    plot_filename = q_plots_dir / "state_action_count_heatmap.jpg"
                    plt.savefig(plot_filename)
                    plt.close()
                    logger.info(f"Saved state-action count heatmaps to {plot_filename}")
                else:
                    logger.info(
                        "No data points left after binning for state-action distribution heatmap."
                    )
            else:
                logger.info(
                    "No valid data points to plot for state-action distribution after processing observations."
                )
        else:
            logger.info("No actions recorded to plot state-action distribution.")
    except Exception as e:
        logger.error(
            f"Error during state-action distribution heatmap plotting: {e}",
            exc_info=True,
        )


def plot_trajectories(df, q_plots_dir, logger):
    try:
        plot_df = _prepare_trajectory_df(df)

        if not plot_df.empty:
            # Determine the number of trajectories
            trajectory_ids = plot_df["trajectory_id"].unique()
            num_trajectories = len(trajectory_ids)

            # Determine the number of rows and columns for subplots
            num_cols = 3  # You can adjust this number based on your preference
            num_rows = (
                num_trajectories + num_cols - 1
            ) // num_cols  # Ensure enough rows

            # Create subplots
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows)
            )
            fig.suptitle("Trajectories", fontsize=16)

            # Flatten the axes array for easy indexing
            axes = axes.flatten()

            # Plot each trajectory in a separate subplot
            for i, traj_id in enumerate(trajectory_ids):
                ax = axes[i]
                traj_df = plot_df[plot_df["trajectory_id"] == traj_id]
                _plot_single_trajectory_on_ax(ax, traj_df)

                traj_name = (
                    traj_df["trajectory_name"].iloc[0]
                    if "trajectory_name" in traj_df.columns
                    else None
                )

                # Remove common prefix from all trajectory names
                all_names = plot_df["trajectory_name"].dropna().unique()
                if len(all_names) > 1:
                    # Find common prefix
                    all_names_list = all_names.tolist()
                    common_prefix = os.path.commonprefix(all_names_list)
                    if common_prefix:
                        # Remove common prefix from this name
                        if traj_name and traj_name.startswith(common_prefix):
                            traj_name = traj_name[len(common_prefix) :]
                        # Also strip leading slashes or underscores
                        traj_name = traj_name.lstrip("/_-") if traj_name else traj_name

                num_segments = len(traj_df)
                actions = (
                    traj_df["action"].values
                    if "action" in traj_df.columns
                    else np.zeros(num_segments)
                )

                # Calculate empirical % action 1
                n_action_1 = np.sum(actions == 1)
                pct_action_1 = (
                    100 * n_action_1 / len(actions) if len(actions) > 0 else 0
                )

                # Add 11x11 grid
                grid_lines = np.linspace(0, 1, 12)
                for x in grid_lines:
                    ax.axvline(
                        x, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
                    )
                for y in grid_lines:
                    ax.axhline(
                        y, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
                    )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("s[0]")
                ax.set_ylabel("s[1]")
                # Title: dataset name and empirical % action 1
                if traj_name:
                    ax.set_title(f"{traj_name}\n% action 1: {pct_action_1:.1f}%")
                else:
                    ax.set_title(f"% action 1: {pct_action_1:.1f}%")

            # Remove any unused subplots
            for i in range(num_trajectories, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout(
                rect=(0, 0.03, 1, 0.95)
            )  # Adjust layout to make space for suptitle
            plot_filename = q_plots_dir / "state_trajectories_subplots.jpg"
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved state trajectories plot to {plot_filename}")
        else:
            logger.info("No valid data points to plot for state trajectories.")
    except Exception as e:
        logger.error(f"Error during state trajectories plotting: {e}", exc_info=True)
        logger.error(f"Error during state trajectories plotting: {e}", exc_info=True)
