import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(os.getcwd())
import argparse
import logging
import random
import socket
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt  # Added import
import numpy as np
import seaborn as sns
import torch
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity, Ignore
from PyExpUtils.results.sqlite import saveCollector

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import expand
from utils.plotting import get_Q, plot_q, plot_q_diff  # Added import
from utils.preempt import TimeoutHandler
from utils.RlGlue.rl_glue import LoggingRlGlue

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True)
parser.add_argument("-i", "--idxs", nargs="+", type=int, required=True)
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--gpu", action="store_true", default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------


device = "gpu" if args.gpu else "cpu"
jax.config.update("jax_platform_name", device)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("exp")
prod = "cdr" in socket.gethostname() or args.silent
if not prod:
    logger.setLevel(logging.DEBUG)

# ----------------------
# -- Experiment Def'n --
# ----------------------
timeout_handler = TimeoutHandler()

exp = ExperimentModel.load(args.exp)
indices = args.idxs

Problem = getProblem(exp.problem)


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


for idx in indices:
    chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
    chk.load_if_exists()
    timeout_handler.before_cancel(chk.save)

    collector = chk.build(
        "collector",
        lambda: Collector(
            # specify which keys to actually store and ultimately save
            # Options are:
            #  - Identity() (save everything)
            #  - Window(n)  take a window average of size n
            #  - Subsample(n) save one of every n elements
            config={
                "return": Identity(),  # total reward at the end of episode
                "reward": Identity(),  # reward at each step
                "episode": Identity(),
                "steps": Identity(),
                "action": Identity(),
            },
            default=Ignore(),
        ),
    )
    collector.setIdx(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly, with optional offset
    params = exp.get_hypers(idx)
    exp_params = params.get("experiment", {})
    seed = run + exp_params.get("seed_offset", 0)

    # Seed various modules
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # build stateful things and attach to checkpoint
    problem = chk.build("p", lambda: Problem(exp, idx, collector))
    agent = chk.build("a", problem.getAgent)
    env = chk.build("e", problem.getEnvironment)

    glue = chk.build("glue", lambda: LoggingRlGlue(agent, env))
    chk.initial_value("episode", 0)

    context = exp.buildSaveContext(idx, base=args.save_path)
    Path(context.resolve()).mkdir(parents=True, exist_ok=True)
    agent_path = Path(context.resolve()).relative_to("results")

    # Create directory for Q-value plots
    q_plots_dir = Path(context.resolve()) / "q_value_plots"
    q_plots_dir.mkdir(parents=True, exist_ok=True)

    config = {**problem.params, "context": str(agent_path)}

    wandb_run = wandb.init(
        entity="plant-rl",
        project="offline",
        notes=str(agent_path),
        config=config,
        settings=wandb.Settings(
            x_stats_disk_paths=("/", "/data"),
        ),
    )

    # Run the experiment
    start_time = time.time()

    data = defaultdict(list)

    # if we haven't started yet, then make the first interaction
    data_exhausted = False
    if glue.total_steps == 0:
        s, env_info = env.start()
        agent.load_start(s, env_info)
        data_exhausted = env_info.get("exhausted", False)
        data["observation"].append(s)
        data["action"].append(None)
        data["reward"].append(None)
        data["terminal"].append(None)
        data["trajectory_name"].append(env_info.get("trajectory_name", None))

    while not data_exhausted:
        (reward, s, term, env_info) = env.step(None)
        data["observation"].append(s)
        data["action"].append(env_info.get("action", None))
        data["reward"].append(reward)
        data["terminal"].append(term)
        data["trajectory_name"].append(env_info.get("trajectory_name", None))
        data_exhausted = env_info.get("exhausted", False)
        if term:
            agent.load_end(reward, env_info)
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()
            if not data_exhausted:
                s, env_info = env.start()
                agent.load_start(s, env_info)
                data["observation"].append(s)
                data["action"].append(None)
                data["reward"].append(None)
                data["terminal"].append(None)
                data["trajectory_name"].append(env_info.get("trajectory_name", None))
                data_exhausted = env_info.get("exhausted", False)
        else:
            agent.load_step(reward, s, env_info)

    df = pd.DataFrame(data)
    df.to_csv(context.resolve("data.csv"), index=False)

    # Plot state-action distribution
    plot_state_action_distribution(df, q_plots_dir, logger)

    # Plot trajectories
    plot_trajectories(df, q_plots_dir, logger)

    for step in range(exp.total_steps):
        info = agent.plan()
        if step % 1000 == 0:
            expanded_info = {}
            for key, value in info.items():
                expanded_info.update(expand(key, value))
            wandb_run.log(expanded_info, step=step)

        if (
            step == 0
            or step % 10 ** int(np.log10(step)) == 0
            or step == exp.total_steps - 1
        ):
            # Plot and save Q-values
            plot_q_values_and_diff(
                logger, agent, q_plots_dir, step, df
            )  # Updated function call

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
