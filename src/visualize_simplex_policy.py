#!/usr/bin/env python3
"""
Visualize simplex policy on a real trajectory from the dataset.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import minari
import numpy as np
import seaborn as sns
import ternary
from flax import nnx

from algorithms.nn.inac.agent.base import load
from algorithms.nn.inac.agent.in_sample import ActorCritic
from utils.metrics import UnbiasedExponentialMovingAverage

matplotlib.use("Agg")
sns.set_style("white")


def load_model(
    exp_path: Path,
    state_dim: int,
    action_dim: int,
    hidden_units: int,
    policy_type: str,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
):
    """Load trained model from checkpoint."""
    import optax

    from algorithms.nn.inac.agent.in_sample import ActorCritic, Optimizers

    rngs = nnx.Rngs(0)
    actor_critic = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_units=hidden_units,
        discrete_control=False,
        policy_type=policy_type,
        rngs=rngs,
    )

    optimizers = Optimizers(
        pi=nnx.Optimizer(
            actor_critic.pi,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        q=nnx.Optimizer(
            actor_critic.q,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        value=nnx.Optimizer(
            actor_critic.value_net,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        beh_pi=nnx.Optimizer(
            actor_critic.beh_pi,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
    )

    # Load parameters
    parameters_dir = exp_path / "parameters"
    if not parameters_dir.exists():
        raise FileNotFoundError(f"Parameters directory not found: {parameters_dir}")

    # Use load to restore parameters - it returns the full actor_critic module
    actor_critic, optimizers = load(
        nnx.split(actor_critic)[0],  # graphdef
        nnx.split(optimizers)[0],  # optimizers graphdef
        parameters_dir,
    )

    return actor_critic


def get_trajectory_actions(episode, pi, beh_pi, rngs):
    """Get policy actions for each state in the trajectory."""
    states = episode.observations[
        :-1
    ]  # Remove last observation (no action for terminal)
    dataset_actions = episode.actions

    # Get policy actions
    policy_actions, _ = pi(states, deterministic=True, rngs=rngs)

    # Get behavior policy actions
    beh_policy_actions, _ = beh_pi(states, deterministic=True, rngs=rngs)

    return states, dataset_actions, policy_actions, beh_policy_actions


def plot_trajectory_actions(
    states,
    dataset_actions,
    policy_actions,
    beh_policy_actions,
    episode_idx,
    save_path=None,
):
    """Plot the actions over the trajectory."""
    time_steps = np.arange(len(states))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Dataset actions
    axes[0].plot(
        time_steps, dataset_actions[:, 0], "r-", label="Red (dataset)", linewidth=2
    )
    axes[0].plot(
        time_steps, dataset_actions[:, 1], "gray", label="White (dataset)", linewidth=2
    )
    axes[0].plot(
        time_steps, dataset_actions[:, 2], "b-", label="Blue (dataset)", linewidth=2
    )
    axes[0].set_title(f"Episode {episode_idx} - Dataset Actions")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Action Coefficient")
    axes[0].legend()
    axes[0].grid(True)

    # Policy actions
    axes[1].plot(time_steps, policy_actions[:, 0], "r--", label="Red (pi)", linewidth=2)
    axes[1].plot(
        time_steps, policy_actions[:, 1], "k--", label="White (pi)", linewidth=2
    )
    axes[1].plot(
        time_steps, policy_actions[:, 2], "b--", label="Blue (pi)", linewidth=2
    )
    axes[1].set_title(f"Episode {episode_idx} - Policy (pi) Actions")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Action Coefficient")
    axes[1].legend()
    axes[1].grid(True)

    # Behavior policy actions
    axes[2].plot(
        time_steps, beh_policy_actions[:, 0], "r:", label="Red (beh_pi)", linewidth=2
    )
    axes[2].plot(
        time_steps, beh_policy_actions[:, 1], "k:", label="White (beh_pi)", linewidth=2
    )
    axes[2].plot(
        time_steps, beh_policy_actions[:, 2], "b:", label="Blue (beh_pi)", linewidth=2
    )
    axes[2].set_title(f"Episode {episode_idx} - Behavior Policy (beh_pi) Actions")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Action Coefficient")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_action_comparison(
    dataset_actions, policy_actions, beh_policy_actions, episode_idx, save_path=None
):
    """Plot comparison of dataset vs policy actions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ["red", "gray", "blue"]
    labels = ["Red", "White", "Blue"]

    # Dataset vs pi
    for i in range(3):
        ax = axes[0, i]
        ax.scatter(
            dataset_actions[:, i], policy_actions[:, i], alpha=0.6, color=colors[i]
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Diagonal line
        ax.set_xlabel(f"Dataset {labels[i]}")
        ax.set_ylabel(f"Policy (pi) {labels[i]}")
        ax.set_title(f"{labels[i]} - Dataset vs pi")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Dataset vs beh_pi
    for i in range(3):
        ax = axes[1, i]
        ax.scatter(
            dataset_actions[:, i], beh_policy_actions[:, i], alpha=0.6, color=colors[i]
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Diagonal line
        ax.set_xlabel(f"Dataset {labels[i]}")
        ax.set_ylabel(f"Behavior Policy (beh_pi) {labels[i]}")
        ax.set_title(f"{labels[i]} - Dataset vs beh_pi")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle(f"Episode {episode_idx} - Action Comparisons")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()


def plot_episode_comprehensive(
    actor_critic,
    states,
    dataset_actions,
    policy_actions,
    beh_policy_actions,
    episode_idx,
    tau=1.0,
    eps=1e-8,
    exp_threshold=10000,
    save_path=None,
):
    """
    Create a comprehensive figure showing all metrics over time for an episode.

    Rows (from top to bottom):
    1. State (all dimensions)
    2. Dataset Actions (RGB)
    3. Q-values (for dataset actions)
    4. Value
    5. Advantage
    6. Clipped Advantage
    7. Behavior Policy log prob (for dataset actions)
    8. Policy log prob (for dataset actions)
    9. Behavior Policy Actions (RGB)
    10. Policy Actions (RGB)
    """
    time_steps = np.arange(len(states))

    # Convert to JAX arrays
    states_jax = jnp.array(states)
    dataset_actions_jax = jnp.array(dataset_actions)

    # Compute all metrics over the trajectory
    q_values, _, _ = actor_critic.q(states_jax, dataset_actions_jax)
    values = actor_critic.value_net(states_jax).squeeze(-1)
    advantages = q_values - values

    # Compute clipped advantages
    beh_log_probs = actor_critic.beh_pi.get_logprob(states_jax, dataset_actions_jax)
    clipped_advantages = jnp.clip(
        jnp.exp((q_values - values) / tau - beh_log_probs),
        eps,
        exp_threshold,
    )

    # Compute policy log probs for dataset actions
    pi_log_probs = actor_critic.pi.get_logprob(states_jax, dataset_actions_jax)

    # Convert to numpy for plotting
    q_values = np.array(q_values)
    values = np.array(values)
    advantages = np.array(advantages)
    clipped_advantages = np.array(clipped_advantages)
    beh_log_probs = np.array(beh_log_probs)
    pi_log_probs = np.array(pi_log_probs)

    # Create figure with 10 subplots
    fig, axes = plt.subplots(10, 1, figsize=(14, 22))

    # 1. States
    for i in range(states.shape[1]):
        axes[0].plot(time_steps, states[:, i], label=f"State dim {i}", alpha=0.7)
    axes[0].set_title(f"Episode {episode_idx} - States")
    axes[0].set_ylabel("State Value")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. Dataset Actions
    axes[1].plot(time_steps, dataset_actions[:, 0], "r-", label="Red", linewidth=2)
    axes[1].plot(
        time_steps,
        dataset_actions[:, 1],
        color="gray",
        linestyle="-",
        label="White",
        linewidth=2,
    )
    axes[1].plot(time_steps, dataset_actions[:, 2], "b-", label="Blue", linewidth=2)
    axes[1].set_title("Dataset Actions")
    axes[1].set_ylabel("Action Coefficient")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    # 3. Q-values (for dataset actions)
    axes[2].plot(time_steps, q_values, "g-", linewidth=2)
    axes[2].set_title("Q-values (for dataset actions)")
    axes[2].set_ylabel("Q-value")
    axes[2].grid(True, alpha=0.3)

    # 4. Value
    axes[3].plot(time_steps, values, "purple", linewidth=2)
    axes[3].set_title("Value Function V(s)")
    axes[3].set_ylabel("Value")
    axes[3].grid(True, alpha=0.3)

    # 5. Advantage
    axes[4].plot(time_steps, advantages, "orange", linewidth=2)
    axes[4].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[4].set_title("Advantage (Q - V)")
    axes[4].set_ylabel("Advantage")
    axes[4].grid(True, alpha=0.3)

    # 6. Clipped Advantage
    axes[5].plot(time_steps, clipped_advantages, "brown", linewidth=2)
    axes[5].set_title(
        f"Clipped Advantage (τ={tau}, eps={eps}, threshold={exp_threshold})"
    )
    axes[5].set_ylabel("Clipped Advantage")
    axes[5].grid(True, alpha=0.3)
    axes[5].set_yscale("log")

    # 7. Behavior Policy log prob
    axes[6].plot(time_steps, beh_log_probs, "cyan", linewidth=2)
    axes[6].set_title("Behavior Policy log π_β(a|s) (for dataset actions)")
    axes[6].set_ylabel("log prob")
    axes[6].grid(True, alpha=0.3)

    # 8. Policy log prob
    axes[7].plot(time_steps, pi_log_probs, "magenta", linewidth=2)
    axes[7].set_title("Policy log π(a|s) (for dataset actions)")
    axes[7].set_ylabel("log prob")
    axes[7].grid(True, alpha=0.3)

    # 9. Behavior Policy Actions
    axes[8].plot(time_steps, beh_policy_actions[:, 0], "r:", label="Red", linewidth=2)
    axes[8].plot(
        time_steps,
        beh_policy_actions[:, 1],
        color="gray",
        linestyle=":",
        label="White",
        linewidth=2,
    )
    axes[8].plot(time_steps, beh_policy_actions[:, 2], "b:", label="Blue", linewidth=2)
    axes[8].set_title("Behavior Policy Actions π_β(s)")
    axes[8].set_ylabel("Action Coefficient")
    axes[8].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[8].grid(True, alpha=0.3)
    axes[8].set_ylim(-0.05, 1.05)

    # 10. Policy Actions
    axes[9].plot(time_steps, policy_actions[:, 0], "r--", label="Red", linewidth=2)
    axes[9].plot(
        time_steps,
        policy_actions[:, 1],
        color="gray",
        linestyle="--",
        label="White",
        linewidth=2,
    )
    axes[9].plot(time_steps, policy_actions[:, 2], "b--", label="Blue", linewidth=2)
    axes[9].set_title("Policy Actions π(s)")
    axes[9].set_xlabel("Time Step")
    axes[9].set_ylabel("Action Coefficient")
    axes[9].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[9].grid(True, alpha=0.3)
    axes[9].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comprehensive episode plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_episode_comprehensive_ternary(
    actor_critic,
    states,
    dataset_actions,
    policy_actions,
    beh_policy_actions,
    episode_idx,
    tau=1.0,
    eps=1e-8,
    exp_threshold=10000,
    num_points=32,
    save_path=None,
):
    """
    Create a comprehensive figure showing all metrics over time for an episode.
    Uses ternary plots for spatial metrics (Q, advantage, value, clipped advantage, policies)
    and line plots for temporal metrics (states, actions, log probs).

    Layout:
    - Row 0: States (time series)
    - Row 1: Dataset Actions, Behavior Policy Actions, Policy Actions (time series), Q&V
    - Row 2: Q-values (ternary), Advantage (ternary), Value (ternary), Clipped Advantage (ternary)
    - Row 3: Behavior Policy PDF (ternary), Policy PDF (ternary), Log probs (time series), Advs (time series)
    """
    time_steps = np.arange(len(states))

    # Convert to JAX arrays
    states_jax = jnp.array(states)
    dataset_actions_jax = jnp.array(dataset_actions)

    # Compute all metrics over the trajectory
    q_values, _, _ = actor_critic.q(states_jax, dataset_actions_jax)
    values = actor_critic.value_net(states_jax).squeeze(-1)
    advantages = q_values - values

    # Compute clipped advantages
    beh_log_probs = actor_critic.beh_pi.get_logprob(states_jax, dataset_actions_jax)
    clipped_advantages = jnp.clip(
        jnp.exp((q_values - values) / tau - beh_log_probs),
        eps,
        exp_threshold,
    )

    # Compute policy log probs for dataset actions
    pi_log_probs = actor_critic.pi.get_logprob(states_jax, dataset_actions_jax)

    # Convert to numpy for plotting
    q_values_np = np.array(q_values)
    values_np = np.array(values)
    advantages_np = np.array(advantages)
    clipped_advantages_np = np.array(clipped_advantages)
    beh_log_probs_np = np.array(beh_log_probs)
    pi_log_probs_np = np.array(pi_log_probs)

    # Create figure with mixed subplot layout
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

    # Row 0: States (spans all columns)
    ax_states = fig.add_subplot(gs[0, :])
    for i in range(states.shape[1]):
        ax_states.plot(time_steps, states[:, i], label=f"State dim {i}", alpha=0.7)
    ax_states.set_title(f"Episode {episode_idx} - States")
    ax_states.set_ylabel("State Value")
    ax_states.legend(loc="upper right", ncol=states.shape[1], fontsize=8)
    ax_states.grid(True, alpha=0.3)

    # Row 1: Actions (3 time series plots)
    # Dataset Actions
    ax_dataset = fig.add_subplot(gs[1, 0])
    ax_dataset.plot(time_steps, dataset_actions[:, 0], "r-", label="R", linewidth=1.5)
    ax_dataset.plot(
        time_steps, dataset_actions[:, 1], color="gray", label="W", linewidth=1.5
    )
    ax_dataset.plot(time_steps, dataset_actions[:, 2], "b-", label="B", linewidth=1.5)
    ax_dataset.set_title("Dataset Actions")
    ax_dataset.set_ylabel("Coefficient")
    ax_dataset.legend(fontsize=8)
    ax_dataset.grid(True, alpha=0.3)
    ax_dataset.set_ylim(-0.05, 1.05)

    # Behavior Policy Actions
    ax_beh = fig.add_subplot(gs[1, 1])
    ax_beh.plot(time_steps, beh_policy_actions[:, 0], "r:", label="R", linewidth=1.5)
    ax_beh.plot(
        time_steps,
        beh_policy_actions[:, 1],
        color="gray",
        linestyle=":",
        label="W",
        linewidth=1.5,
    )
    ax_beh.plot(time_steps, beh_policy_actions[:, 2], "b:", label="B", linewidth=1.5)
    ax_beh.set_title("Behavior Policy Actions π_β(s)")
    ax_beh.set_ylabel("Coefficient")
    ax_beh.legend(fontsize=8)
    ax_beh.grid(True, alpha=0.3)
    ax_beh.set_ylim(-0.05, 1.05)

    # Policy Actions
    ax_pi = fig.add_subplot(gs[1, 2])
    ax_pi.plot(time_steps, policy_actions[:, 0], "r--", label="R", linewidth=1.5)
    ax_pi.plot(
        time_steps,
        policy_actions[:, 1],
        color="gray",
        linestyle="--",
        label="W",
        linewidth=1.5,
    )
    ax_pi.plot(time_steps, policy_actions[:, 2], "b--", label="B", linewidth=1.5)
    ax_pi.set_title("Policy Actions π(s)")
    ax_pi.set_ylabel("Coefficient")
    ax_pi.legend(fontsize=8)
    ax_pi.grid(True, alpha=0.3)
    ax_pi.set_ylim(-0.05, 1.05)

    # Q-values and Value combined plot
    ax_qv = fig.add_subplot(gs[1, 3])
    ax_qv.plot(time_steps, q_values_np, "g-", linewidth=1.5, label="Q(s,a)")
    ax_qv.plot(time_steps, values_np, "purple", linewidth=1.5, label="V(s)")
    ax_qv.set_title("Q-values & Value")
    ax_qv.set_ylabel("Value")
    ax_qv.legend(fontsize=8)
    ax_qv.grid(True, alpha=0.3)

    # Row 2: Ternary plots for Q, Advantage, Value, Clipped Advantage
    # Use middle timestep as representative state
    mid_idx = len(states) // 2
    state_mid = states[mid_idx]

    # Generate simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])
    points = np.array(points, dtype=float)
    simplex_actions = points / num_points

    # Compute metrics over simplex for middle state
    simplex_actions_jax = jnp.array(simplex_actions)
    state_mid_jax = jnp.array(state_mid).reshape(1, -1)
    state_mid_tiled = jnp.tile(state_mid_jax, (simplex_actions_jax.shape[0], 1))

    q_simplex, _, _ = actor_critic.q(state_mid_tiled, simplex_actions_jax)
    value_mid = actor_critic.value_net(state_mid_jax).squeeze(-1)
    adv_simplex = q_simplex - value_mid

    beh_log_prob_simplex = actor_critic.beh_pi.get_logprob(
        state_mid_tiled, simplex_actions_jax
    )
    clipped_adv_simplex = jnp.clip(
        jnp.exp((q_simplex - value_mid) / tau - beh_log_prob_simplex),
        eps,
        exp_threshold,
    )

    pi_log_prob_simplex = actor_critic.pi.get_logprob(
        state_mid_tiled, simplex_actions_jax
    )

    # Q-values ternary
    ax_q_tern = fig.add_subplot(gs[2, 0])
    plot_ternary_on_axis(
        ax_q_tern,
        points,
        np.array(q_simplex),
        num_points,
        f"Q (t={mid_idx})",
        "viridis",
    )

    # Advantage ternary
    ax_adv_tern = fig.add_subplot(gs[2, 1])
    plot_ternary_on_axis(
        ax_adv_tern,
        points,
        np.array(adv_simplex),
        num_points,
        f"Adv (t={mid_idx})",
        "RdBu_r",
    )

    # Value ternary (constant)
    ax_val_tern = fig.add_subplot(gs[2, 2])
    value_constant = np.full(len(points), float(value_mid[0]))
    plot_ternary_on_axis(
        ax_val_tern,
        points,
        value_constant,
        num_points,
        f"V={float(value_mid[0]):.2f} (t={mid_idx})",
        "viridis",
    )

    # Clipped Advantage ternary
    ax_cadv_tern = fig.add_subplot(gs[2, 3])
    plot_ternary_on_axis(
        ax_cadv_tern,
        points,
        np.array(clipped_adv_simplex),
        num_points,
        f"Clip Adv (t={mid_idx})",
        "inferno",
    )

    # Row 3: Policy PDFs and log probs
    # Behavior Policy PDF ternary
    ax_beh_pdf_tern = fig.add_subplot(gs[3, 0])
    beh_log10_pdf = np.array(beh_log_prob_simplex) * np.log10(np.e)
    plot_ternary_on_axis(
        ax_beh_pdf_tern,
        points,
        beh_log10_pdf,
        num_points,
        f"π_β log10 (t={mid_idx})",
        "plasma",
    )

    # Policy PDF ternary
    ax_pi_pdf_tern = fig.add_subplot(gs[3, 1])
    pi_log10_pdf = np.array(pi_log_prob_simplex) * np.log10(np.e)
    plot_ternary_on_axis(
        ax_pi_pdf_tern,
        points,
        pi_log10_pdf,
        num_points,
        f"π log10 (t={mid_idx})",
        "plasma",
    )

    # Log probs time series
    ax_logprobs = fig.add_subplot(gs[3, 2])
    ax_logprobs.plot(
        time_steps, beh_log_probs_np, "cyan", linewidth=1.5, label="log π_β(a|s)"
    )
    ax_logprobs.plot(
        time_steps, pi_log_probs_np, "magenta", linewidth=1.5, label="log π(a|s)"
    )
    ax_logprobs.set_title("Log Probs (dataset actions)")
    ax_logprobs.set_xlabel("Time Step")
    ax_logprobs.set_ylabel("log prob")
    ax_logprobs.legend(fontsize=8)
    ax_logprobs.grid(True, alpha=0.3)

    # Advantage and Clipped Advantage time series
    ax_advs = fig.add_subplot(gs[3, 3])
    ax_advs.plot(time_steps, advantages_np, "orange", linewidth=1.5, label="Advantage")
    ax_advs.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax_advs.set_ylabel("Advantage", color="orange")
    ax_advs.tick_params(axis="y", labelcolor="orange")
    ax_advs.grid(True, alpha=0.3)

    ax_advs2 = ax_advs.twinx()
    ax_advs2.plot(
        time_steps, clipped_advantages_np, "brown", linewidth=1.5, label="Clipped Adv"
    )
    ax_advs2.set_ylabel("Clipped Adv (log)", color="brown")
    ax_advs2.set_yscale("log")
    ax_advs2.tick_params(axis="y", labelcolor="brown")

    ax_advs.set_title("Advantages")
    ax_advs.set_xlabel("Time Step")
    ax_advs.legend(loc="upper left", fontsize=8)
    ax_advs2.legend(loc="upper right", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comprehensive episode plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ternary_on_axis(
    ax, points, values, num_points, title, cmap="viridis", vmin=None, vmax=None
):
    """Helper function to plot a ternary heatmap on a given axis."""
    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Create ternary plot
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Create data dict
    data = {}
    for point, val in zip(points, values, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(val)

    # Plot heatmap. Pass vmin/vmax if provided so multiple subplots can share scale.
    heatmap_kwargs = dict(
        style="hexagonal", cmap=plt.get_cmap(cmap), colorbar=True, cbarlabel=""
    )
    if vmin is not None:
        heatmap_kwargs["vmin"] = float(vmin)  # type: ignore
    if vmax is not None:
        heatmap_kwargs["vmax"] = float(vmax)  # type: ignore

    tax.heatmap(data, **heatmap_kwargs)  # type: ignore

    # Add corner labels
    tax.right_corner_label("R", fontsize=10)
    tax.top_corner_label("W", fontsize=10)
    tax.left_corner_label("B", fontsize=10)

    # Set title
    tax.set_title(title, fontsize=11, pad=20)


def plot_episode_ternary_timeseries(
    actor_critic,
    states,
    dataset_actions,
    rewards,
    episode_idx,
    rngs,
    tau=1.0,
    eps=1e-8,
    exp_threshold=10000,
    num_points=24,
    num_timesteps=8,
    save_path=None,
):
    """
    Create a grid of ternary plots showing how metrics evolve over time.

    Columns: Timesteps (evenly sampled from the episode)
    Rows: Different metrics
    - Row 0: Q-values (with dataset action marker)
    - Row 1: Advantage (with dataset action marker)
    - Row 2: Value (constant per state)
    - Row 3: Clipped Advantage (with dataset action marker)
    - Row 4: Behavior Policy log10 PDF (with mean and mode markers)
    - Row 5: Policy log10 PDF (with mean and mode markers)

    Args:
        actor_critic: The trained actor-critic model
        states: Episode states
        dataset_actions: Actions taken in the dataset
        rewards: Rewards received
        episode_idx: Episode index for labeling
        rngs: Random number generators
        tau: Temperature parameter
        eps: Lower clipping threshold
        exp_threshold: Upper clipping threshold
        num_points: Resolution of simplex grid
        num_timesteps: Number of timesteps to show
        save_path: Path to save figure
    """
    # Select evenly spaced timesteps
    total_steps = len(states)
    if num_timesteps > total_steps:
        num_timesteps = total_steps
    timestep_indices = np.linspace(0, total_steps - 1, num_timesteps, dtype=int)

    # Generate simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])
    points = np.array(points, dtype=float)
    simplex_actions = points / num_points
    simplex_actions_jax = jnp.array(simplex_actions)

    # Create figure with grid layout
    # 6 rows (metrics) x num_timesteps columns
    fig = plt.figure(figsize=(3 * num_timesteps, 18))
    gs = fig.add_gridspec(6, num_timesteps, hspace=0.3, wspace=0.3)

    metric_names = [
        "Q-values",
        "Advantage",
        "Value",
        "Clipped Adv",
        "π_β log10 PDF",
        "π log10 PDF",
    ]
    cmaps = ["viridis", "RdBu_r", "viridis", "inferno", "plasma", "plasma"]

    # First pass: compute metrics for all selected timesteps so we can derive
    # a consistent color scale (vmin/vmax) per row across columns.
    per_timestep_data = []
    for t in timestep_indices:
        state = states[t]
        state_jax = jnp.array(state).reshape(1, -1)
        state_tiled = jnp.tile(state_jax, (simplex_actions_jax.shape[0], 1))

        # Get dataset action for this timestep
        dataset_action = dataset_actions[t]

        # Compute mean actions from policies
        beh_mean_action, _ = actor_critic.beh_pi(
            state_jax, deterministic=True, rngs=rngs
        )
        pi_mean_action, _ = actor_critic.pi(state_jax, deterministic=True, rngs=rngs)
        beh_mean_action = np.array(beh_mean_action[0])
        pi_mean_action = np.array(pi_mean_action[0])

        # Compute all metrics for this state
        q_values, _, _ = actor_critic.q(state_tiled, simplex_actions_jax)
        value = actor_critic.value_net(state_jax).squeeze(-1)
        advantages = q_values - value

        beh_log_probs = actor_critic.beh_pi.get_logprob(
            state_tiled, simplex_actions_jax
        )
        clipped_advantages = jnp.clip(
            jnp.exp((q_values - value) / tau - beh_log_probs), eps, exp_threshold
        )

        pi_log_probs = actor_critic.pi.get_logprob(state_tiled, simplex_actions_jax)

        # Find mode (action with highest probability) for each policy
        beh_probs = np.exp(np.array(beh_log_probs))
        pi_probs = np.exp(np.array(pi_log_probs))
        beh_mode_idx = np.argmax(beh_probs)
        pi_mode_idx = np.argmax(pi_probs)
        beh_mode_action = simplex_actions[beh_mode_idx]
        pi_mode_action = simplex_actions[pi_mode_idx]

        # Convert to numpy and create metric list
        metrics = [
            np.array(q_values),
            np.array(advantages),
            np.full(len(points), float(value[0])),  # Constant value
            np.array(clipped_advantages),
            np.array(beh_log_probs) * np.log10(np.e),
            np.array(pi_log_probs) * np.log10(np.e),
        ]

        mean_actions = [
            dataset_action,
            dataset_action,
            None,
            dataset_action,
            beh_mean_action,
            pi_mean_action,
        ]

        mode_actions = [
            None,
            None,
            None,
            None,
            beh_mode_action,
            pi_mode_action,
        ]

        per_timestep_data.append(
            {
                "t": int(t),
                "metrics": metrics,
                "mean_actions": mean_actions,
                "mode_actions": mode_actions,
            }
        )

    # Compute vmin/vmax per metric (row) across all timesteps (columns)
    n_metrics = len(cmaps)
    vmins = [None] * n_metrics
    vmaxs = [None] * n_metrics
    for m_idx in range(n_metrics):
        vals = [pd["metrics"][m_idx] for pd in per_timestep_data]
        # Stack to compute global min/max
        stacked = np.stack([np.array(v) for v in vals], axis=0)
        vmins[m_idx] = float(np.nanmin(stacked))  # type: ignore
        vmaxs[m_idx] = float(np.nanmax(stacked))  # type: ignore

    # Second pass: plotting using shared vmin/vmax per row
    for col_idx, pd in enumerate(per_timestep_data):
        t = pd["t"]
        metrics = pd["metrics"]
        mean_actions = pd["mean_actions"]
        mode_actions = pd["mode_actions"]

        for row_idx, (metric_data, cmap, mean_action, mode_action) in enumerate(
            zip(metrics, cmaps, mean_actions, mode_actions, strict=True)
        ):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Add timestep to title only for top row
            title = f"t={t}" if row_idx == 0 else ""

            plot_ternary_on_axis(
                ax,
                points,
                metric_data,
                num_points,
                title,
                cmap,
                vmin=vmins[row_idx],
                vmax=vmaxs[row_idx],
            )

            # Add markers if applicable
            if mean_action is not None or mode_action is not None:
                figure, tax = ternary.figure(ax=ax, scale=num_points)

                if mean_action is not None:
                    action_ternary = tuple(mean_action * num_points)
                    if row_idx in [4, 5]:  # Policy mean actions
                        tax.scatter(
                            [action_ternary],
                            marker="*",
                            s=200,
                            facecolors="none",
                            edgecolors="red",
                            linewidths=2,
                            zorder=10,
                        )
                    else:
                        tax.scatter(
                            [action_ternary],
                            marker="o",
                            s=120,
                            facecolors="none",
                            edgecolors="yellow",
                            linewidths=2,
                            alpha=0.8,
                            zorder=10,
                        )

                if mode_action is not None:
                    mode_ternary = tuple(mode_action * num_points)
                    tax.scatter(
                        [mode_ternary],
                        marker="D",
                        s=120,
                        facecolors="none",
                        edgecolors="cyan",
                        linewidths=2,
                        alpha=0.8,
                        zorder=11,
                    )

    # Add row labels on the left side
    for row_idx, metric_name in enumerate(metric_names):
        # Add text outside the leftmost plot
        fig.text(
            0.02,
            0.92 - (row_idx * 0.153),
            metric_name,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
        )

    # Add overall title
    fig.suptitle(
        f"Episode {episode_idx} - Ternary Plot Time Series\n"
        f"(○ = dataset action, ★ = policy mean, ◇ = policy mode)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved ternary timeseries plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_q_ternary(actor_critic, state, num_points=64, save_path=None):
    """Create a ternary plot of Q-values over the simplex for a fixed state using python-ternary."""
    # Generate ALL points on the simplex grid (including edges and corners)
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    # Normalize to sum to 1
    points = np.array(points, dtype=float)
    actions = points / num_points  # Now sums to 1

    # Convert to JAX array
    actions_jax = jnp.array(actions)
    state_jax = jnp.array(state).reshape(1, -1)
    state_jax = jnp.tile(state_jax, (actions_jax.shape[0], 1))

    # Compute Q-values
    q_values, _, _ = actor_critic.q(state_jax, actions_jax)

    # Create the ternary plot with equilateral triangle aspect ratio
    fig, ax = plt.subplots(
        figsize=(8, 8 * np.sqrt(3) / 2)
    )  # Height adjusted for equilateral triangle

    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set up the ternary plot without boundary
    figure, tax = ternary.figure(ax=ax, scale=num_points)
    # Remove the boundary line
    # tax.boundary(linewidth=2.0)  # Commented out to remove ternary border

    # Create color map
    cmap = plt.get_cmap("viridis")

    # Create data dict for heatmap - use integer coordinates directly
    data = {}
    for point, q_val in zip(points, q_values, strict=True):
        # Keep as integers (no normalization back)
        coord = tuple(point.astype(int))
        data[coord] = float(q_val)

    # Plot heatmap with proper style and colorbar
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Highlight vertices and add labels
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_q = []
    for v in vertices:
        v_norm = v / num_points
        v_jax = jnp.array(v_norm).reshape(1, -1)
        state_tile = jnp.array(state).reshape(1, -1)
        q, _, _ = actor_critic.q(state_tile, v_jax)
        vertex_q.append(float(q[0]))

    print("Q-values at vertices:")
    print(f"Red: {vertex_q[0]:.3f}")
    print(f"White: {vertex_q[1]:.3f}")
    print(f"Blue: {vertex_q[2]:.3f}")

    # Add corner labels with adjusted positioning to avoid title overlap
    tax.right_corner_label(f"Red\n{vertex_q[0]:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{vertex_q[1]:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{vertex_q[2]:.3f}", fontsize=12)

    # Adjust title with more padding to avoid label overlap
    tax.set_title("Q-values over Simplex Action Space", fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_advantage_ternary(actor_critic, state, num_points=64, save_path=None):
    """Create a ternary plot of advantage (Q - V) over the simplex for a fixed state."""
    # Generate ALL points on the simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    points = np.array(points, dtype=float)
    actions = points / num_points

    # Convert to JAX array
    actions_jax = jnp.array(actions)
    state_jax = jnp.array(state).reshape(1, -1)
    state_jax = jnp.tile(state_jax, (actions_jax.shape[0], 1))

    # Compute Q-values and value
    q_values, _, _ = actor_critic.q(state_jax, actions_jax)
    value = actor_critic.value_net(state_jax[0:1]).squeeze(-1)
    advantages = q_values - value

    # Create the ternary plot
    fig, ax = plt.subplots(figsize=(8, 8 * np.sqrt(3) / 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    figure, tax = ternary.figure(ax=ax, scale=num_points)
    cmap = plt.get_cmap("RdBu_r")

    data = {}
    for point, adv_val in zip(points, advantages, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(adv_val)

    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Get vertex values
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_adv = []
    for v in vertices:
        coord = tuple(v.astype(int))
        vertex_adv.append(data[coord])

    tax.right_corner_label(f"Red\n{vertex_adv[0]:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{vertex_adv[1]:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{vertex_adv[2]:.3f}", fontsize=12)
    tax.set_title("Advantage over Simplex Action Space", fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_value_ternary(actor_critic, state, num_points=64, save_path=None):
    """Create a ternary plot showing the value function (constant across actions)."""
    # Generate ALL points on the simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    points = np.array(points, dtype=float)

    # Compute value (independent of action)
    state_jax = jnp.array(state).reshape(1, -1)
    value = actor_critic.value_net(state_jax).squeeze(-1)
    value_scalar = float(value[0])  # Extract the scalar from the batch

    # Create the ternary plot
    fig, ax = plt.subplots(figsize=(8, 8 * np.sqrt(3) / 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    figure, tax = ternary.figure(ax=ax, scale=num_points)
    cmap = plt.get_cmap("viridis")

    # All points have the same value
    data = {}
    for point in points:
        coord = tuple(point.astype(int))
        data[coord] = value_scalar

    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    tax.right_corner_label(f"Red\n{value_scalar:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{value_scalar:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{value_scalar:.3f}", fontsize=12)
    tax.set_title(f"Value Function: V(s) = {value_scalar:.3f}", fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_clipped_advantage_ternary(
    actor_critic,
    state,
    tau=1.0,
    eps=1e-8,
    exp_threshold=10000,
    num_points=64,
    save_path=None,
):
    """Create a ternary plot of the clipped advantage over the simplex for a fixed state."""
    # Generate ALL points on the simplex grid (including edges and corners)
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    # Normalize to sum to 1
    points = np.array(points, dtype=float)
    actions = points / num_points  # Now sums to 1

    # Convert to JAX array
    actions_jax = jnp.array(actions)
    state_jax = jnp.array(state).reshape(1, -1)
    state_jax = jnp.tile(state_jax, (actions_jax.shape[0], 1))

    # Compute components for clipped advantage
    # min_Q
    min_Q, _, _ = actor_critic.q(state_jax, actions_jax)

    # value
    value = actor_critic.value_net(state_jax).squeeze(-1)

    # beh_log_prob
    beh_log_prob = actor_critic.beh_pi.get_logprob(state_jax, actions_jax)

    # Compute clipped advantage
    clipped_advantage = jnp.clip(
        jnp.exp((min_Q - value) / tau - beh_log_prob),
        eps,
        exp_threshold,
    )

    # Create the ternary plot with equilateral triangle aspect ratio
    fig, ax = plt.subplots(
        figsize=(8, 8 * np.sqrt(3) / 2)
    )  # Height adjusted for equilateral triangle

    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set up the ternary plot without boundary
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Create color map
    cmap = plt.get_cmap("inferno")

    # Create data dict for heatmap - use integer coordinates directly
    data = {}
    for point, adv_val in zip(points, clipped_advantage, strict=True):
        # Keep as integers (no normalization back)
        coord = tuple(point.astype(int))
        data[coord] = float(adv_val)

    # Plot heatmap with proper style and colorbar
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Highlight vertices and add labels
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_adv = []
    for v in vertices:
        v_norm = v / num_points
        v_jax = jnp.array(v_norm).reshape(1, -1)
        state_tile = jnp.array(state).reshape(1, -1)

        min_Q, _, _ = actor_critic.q(state_tile, v_jax)
        value = actor_critic.value_net(state_tile).squeeze(-1)
        beh_log_prob = actor_critic.beh_pi.get_logprob(state_tile, v_jax)

        clipped_adv = jnp.clip(
            jnp.exp((min_Q - value) / tau - beh_log_prob),
            eps,
            exp_threshold,
        )
        vertex_adv.append(float(clipped_adv[0]))

    print("Clipped advantage at vertices:")
    print(f"Red: {vertex_adv[0]:.3f}")
    print(f"White: {vertex_adv[1]:.3f}")
    print(f"Blue: {vertex_adv[2]:.3f}")

    # Add corner labels with adjusted positioning to avoid title overlap
    tax.right_corner_label(f"Red\n{vertex_adv[0]:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{vertex_adv[1]:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{vertex_adv[2]:.3f}", fontsize=12)

    # Adjust title with more padding to avoid label overlap
    tax.set_title("Clipped Advantage over Simplex Action Space", fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def compute_clipped_advantage_over_dataset(
    actor_critic,
    dataset,
    tau=1.0,
    eps=1e-8,
    exp_threshold=10000,
    num_points=64,
    max_transitions=None,
):
    """
    Compute the average clipped advantage over the entire dataset.

    For each simplex grid point, computes the clipped advantage across all
    state-action pairs in the dataset and returns the mean.

    Args:
        actor_critic: The trained actor-critic model
        dataset: The Minari dataset
        tau: Temperature parameter
        eps: Lower clipping threshold
        exp_threshold: Upper clipping threshold
        num_points: Number of points per simplex dimension
        max_transitions: Maximum number of transitions to process (None = all)

    Returns:
        Dictionary with keys 'points' (simplex coordinates) and 'clipped_advantages' (mean values)
    """
    # Generate ALL points on the simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    points = np.array(points, dtype=float)
    simplex_actions = points / num_points  # Normalize to sum to 1

    # Collect all states and actions from dataset
    all_states = []
    all_actions = []

    transition_count = 0
    for episode in dataset.iterate_episodes():
        states = episode.observations[:-1]  # Remove last observation
        actions = episode.actions

        all_states.append(states)
        all_actions.append(actions)

        transition_count += len(states)
        if max_transitions is not None and transition_count >= max_transitions:
            break

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    if max_transitions is not None:
        all_states = all_states[:max_transitions]
        all_actions = all_actions[:max_transitions]

    print(f"Computing clipped advantage over {len(all_states)} transitions...")

    # For each simplex point, compute the average clipped advantage
    mean_clipped_advantages = []

    for idx, simplex_action in enumerate(simplex_actions):
        if idx % 100 == 0:
            print(f"  Processing simplex point {idx}/{len(simplex_actions)}...")

        # Tile the simplex action to match all states
        actions_tiled = jnp.tile(simplex_action.reshape(1, -1), (len(all_states), 1))
        states_jax = jnp.array(all_states)

        # Compute components for clipped advantage
        min_Q, _, _ = actor_critic.q(states_jax, actions_tiled)
        value = actor_critic.value_net(states_jax).squeeze(-1)
        beh_log_prob = actor_critic.beh_pi.get_logprob(states_jax, actions_tiled)

        # Compute clipped advantage for all states with this action
        clipped_advantages = jnp.clip(
            jnp.exp((min_Q - value) / tau - beh_log_prob),
            eps,
            exp_threshold,
        )

        # Take the mean across all states
        mean_adv = float(jnp.mean(clipped_advantages))
        mean_clipped_advantages.append(mean_adv)

    return {
        "points": points,
        "clipped_advantages": np.array(mean_clipped_advantages),
        "simplex_actions": simplex_actions,
    }


def plot_dataset_clipped_advantage_ternary(
    advantage_data,
    num_points=64,
    save_path=None,
):
    """
    Create a ternary plot of mean clipped advantage over the dataset.

    Args:
        advantage_data: Dictionary from compute_clipped_advantage_over_dataset
        num_points: Number of points per simplex dimension
        save_path: Path to save the plot
    """
    points = advantage_data["points"]
    clipped_advantages = advantage_data["clipped_advantages"]

    # Create the ternary plot with equilateral triangle aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8 * np.sqrt(3) / 2))

    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set up the ternary plot without boundary
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Create color map
    cmap = plt.get_cmap("inferno")

    # Create data dict for heatmap
    data = {}
    for point, adv_val in zip(points, clipped_advantages, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(adv_val)

    # Plot heatmap with proper style and colorbar
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Get vertex values
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_adv = []
    for v in vertices:
        coord = tuple(v.astype(int))
        vertex_adv.append(data[coord])

    print("\nMean clipped advantage at vertices (over entire dataset):")
    print(f"Red: {vertex_adv[0]:.3f}")
    print(f"White: {vertex_adv[1]:.3f}")
    print(f"Blue: {vertex_adv[2]:.3f}")

    # Add corner labels
    tax.right_corner_label(f"Red\n{vertex_adv[0]:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{vertex_adv[1]:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{vertex_adv[2]:.3f}", fontsize=12)

    # Title
    tax.set_title("Mean Clipped Advantage over Dataset", fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved dataset-level clipped advantage plot to {save_path}")
    else:
        plt.show()

    plt.close()


def compute_dataset_metrics(
    actor_critic,
    dataset,
    tau=1.0,
    eps=1e-8,
    exp_threshold=10000,
    num_points=64,
    max_transitions=None,
):
    """
    Compute various metrics over the entire dataset.

    For each simplex grid point, computes advantage, value, action-value, and clipped advantage
    across all state-action pairs in the dataset and returns the mean.

    Args:
        actor_critic: The trained actor-critic model
        dataset: The Minari dataset
        tau: Temperature parameter for clipped advantage
        eps: Lower clipping threshold for clipped advantage
        exp_threshold: Upper clipping threshold for clipped advantage
        num_points: Number of points per simplex dimension
        max_transitions: Maximum number of transitions to process (None = all)

    Returns:
        Dictionary with keys 'points', 'advantages', 'values', 'action_values', 'clipped_advantages'
    """
    # Generate ALL points on the simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    points = np.array(points, dtype=float)
    simplex_actions = points / num_points  # Normalize to sum to 1

    # Collect all states and actions from dataset
    all_states = []
    all_actions = []

    transition_count = 0
    for episode in dataset.iterate_episodes():
        states = episode.observations[:-1]  # Remove last observation
        actions = episode.actions

        all_states.append(states)
        all_actions.append(actions)

        transition_count += len(states)
        if max_transitions is not None and transition_count >= max_transitions:
            break

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    if max_transitions is not None:
        all_states = all_states[:max_transitions]
        all_actions = all_actions[:max_transitions]

    print(f"Computing metrics over {len(all_states)} transitions...")

    # For each simplex point, compute the average metrics
    mean_advantages = []
    mean_values = []
    mean_action_values = []
    mean_clipped_advantages = []

    for idx, simplex_action in enumerate(simplex_actions):
        if idx % 100 == 0:
            print(f"  Processing simplex point {idx}/{len(simplex_actions)}...")

        # Tile the simplex action to match all states
        actions_tiled = jnp.tile(simplex_action.reshape(1, -1), (len(all_states), 1))
        states_jax = jnp.array(all_states)

        # Compute metrics
        action_value, _, _ = actor_critic.q(states_jax, actions_tiled)
        value = actor_critic.value_net(states_jax).squeeze(-1)
        advantage = action_value - value

        # Compute clipped advantage
        beh_log_prob = actor_critic.beh_pi.get_logprob(states_jax, actions_tiled)
        clipped_advantage = jnp.clip(
            jnp.exp((action_value - value) / tau - beh_log_prob),
            eps,
            exp_threshold,
        )

        # Take the mean across all states
        mean_adv = float(jnp.mean(advantage))
        mean_val = float(jnp.mean(value))
        mean_q = float(jnp.mean(action_value))
        mean_clipped_adv = float(jnp.mean(clipped_advantage))

        mean_advantages.append(mean_adv)
        mean_values.append(mean_val)
        mean_action_values.append(mean_q)
        mean_clipped_advantages.append(mean_clipped_adv)

    return {
        "points": points,
        "advantages": np.array(mean_advantages),
        "values": np.array(mean_values),
        "action_values": np.array(mean_action_values),
        "clipped_advantages": np.array(mean_clipped_advantages),
        "simplex_actions": simplex_actions,
    }


def plot_dataset_metric_ternary(
    metric_data,
    metric_key,
    title,
    num_points=64,
    cmap_name="viridis",
    save_path=None,
):
    """
    Create a ternary plot of mean metric over the dataset.

    Args:
        metric_data: Dictionary from compute_dataset_metrics
        metric_key: Key in metric_data to plot ('advantages', 'values', or 'action_values')
        title: Title for the plot
        num_points: Number of points per simplex dimension
        cmap_name: Name of the colormap to use
        save_path: Path to save the plot
    """
    points = metric_data["points"]
    metric_values = metric_data[metric_key]

    # Create the ternary plot with equilateral triangle aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8 * np.sqrt(3) / 2))

    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set up the ternary plot without boundary
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Create color map
    cmap = plt.get_cmap(cmap_name)

    # Create data dict for heatmap
    data = {}
    for point, val in zip(points, metric_values, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(val)

    # Plot heatmap with proper style and colorbar
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Get vertex values
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_vals = []
    for v in vertices:
        coord = tuple(v.astype(int))
        vertex_vals.append(data[coord])

    print(f"\n{title} at vertices (over entire dataset):")
    print(f"Red: {vertex_vals[0]:.3f}")
    print(f"White: {vertex_vals[1]:.3f}")
    print(f"Blue: {vertex_vals[2]:.3f}")

    # Add corner labels
    tax.right_corner_label(f"Red\n{vertex_vals[0]:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{vertex_vals[1]:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{vertex_vals[2]:.3f}", fontsize=12)

    # Title
    tax.set_title(title, fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {title} plot to {save_path}")
    else:
        plt.show()

    plt.close()


def compute_dataset_policy_metrics(
    policy,
    dataset,
    num_points=64,
    max_transitions=None,
):
    """
    Compute the average policy log probabilities over the entire dataset.

    For each simplex grid point (action), computes the log probability across all
    states in the dataset and returns the mean.

    Args:
        policy: The policy network (pi or beh_pi)
        dataset: The Minari dataset
        num_points: Number of points per simplex dimension
        max_transitions: Maximum number of transitions to process (None = all)

    Returns:
        Dictionary with keys 'points' and 'log_probs' (mean log probabilities)
    """
    # Generate ALL points on the simplex grid
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    points = np.array(points, dtype=float)
    simplex_actions = points / num_points  # Normalize to sum to 1

    # Collect all states from dataset
    all_states = []

    transition_count = 0
    for episode in dataset.iterate_episodes():
        states = episode.observations[:-1]  # Remove last observation

        all_states.append(states)

        transition_count += len(states)
        if max_transitions is not None and transition_count >= max_transitions:
            break

    all_states = np.concatenate(all_states, axis=0)

    if max_transitions is not None:
        all_states = all_states[:max_transitions]

    print(f"Computing policy log probs over {len(all_states)} transitions...")

    # For each simplex point, compute the average log probability
    mean_log_probs = []

    for idx, simplex_action in enumerate(simplex_actions):
        if idx % 100 == 0:
            print(f"  Processing simplex point {idx}/{len(simplex_actions)}...")

        # Tile the simplex action to match all states
        actions_tiled = jnp.tile(simplex_action.reshape(1, -1), (len(all_states), 1))
        states_jax = jnp.array(all_states)

        # Compute log probabilities
        try:
            log_probs = policy.get_logprob(states_jax, actions_tiled)
            mean_lp = float(jnp.mean(log_probs))
        except Exception:
            # Fallback if get_logprob fails
            mean_lp = 0.0

        mean_log_probs.append(mean_lp)

    return {
        "points": points,
        "log_probs": np.array(mean_log_probs),
        "simplex_actions": simplex_actions,
    }


def plot_dataset_policy_ternary(
    policy_data,
    policy_name="Policy",
    num_points=64,
    save_path=None,
):
    """
    Create a ternary plot of mean policy log10 PDF over the dataset.

    Args:
        policy_data: Dictionary from compute_dataset_policy_metrics
        policy_name: Name of the policy for the title
        num_points: Number of points per simplex dimension
        save_path: Path to save the plot
    """
    points = policy_data["points"]
    log_probs = policy_data["log_probs"]

    # Convert to log10 scale
    log10_pdf_values = log_probs * np.log10(np.e)

    # Create the ternary plot with equilateral triangle aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8 * np.sqrt(3) / 2))

    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set up the ternary plot without boundary
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Create color map
    cmap = plt.get_cmap("plasma")

    # Create data dict for heatmap
    data = {}
    for point, pdf_val in zip(points, log10_pdf_values, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(pdf_val)

    # Plot heatmap with proper style and colorbar
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Get vertex values
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_vals = []
    for v in vertices:
        coord = tuple(v.astype(int))
        vertex_vals.append(data[coord])

    print(f"\nMean {policy_name} log10 PDF at vertices (over entire dataset):")
    print(f"Red: {vertex_vals[0]:.3f}")
    print(f"White: {vertex_vals[1]:.3f}")
    print(f"Blue: {vertex_vals[2]:.3f}")

    # Add corner labels
    tax.right_corner_label(f"Red\n{vertex_vals[0]:.3f}", fontsize=12)
    tax.top_corner_label(f"White\n{vertex_vals[1]:.3f}", fontsize=12)
    tax.left_corner_label(f"Blue\n{vertex_vals[2]:.3f}", fontsize=12)

    # Title
    tax.set_title(f"Mean {policy_name} log10 PDF over Dataset", fontsize=14, pad=40)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {policy_name} plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_policy_ternary(
    policy, state, rngs, policy_name="pi", num_points=64, save_path=None
):
    """Create a ternary plot of policy log10 PDF (Probability Density Function) over the simplex for a fixed state."""
    # Generate ALL points on the simplex grid (including edges and corners)
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    # Normalize to sum to 1
    points = np.array(points, dtype=float)
    simplex_points = points / num_points  # Now sums to 1

    # Convert to JAX array
    actions_jax = jnp.array(simplex_points)
    state_jax = jnp.array(state).reshape(1, -1)

    # Compute log prob for each action on the simplex
    # We need to evaluate the policy's probability density at each action
    log_probs = []
    for action in actions_jax:
        action_batch = action.reshape(1, -1)

        # Use the policy's get_logprob method which handles the distribution internally
        try:
            lp = policy.get_logprob(state_jax, action_batch)
            # Convert to scalar
            lp_val = float(jnp.asarray(lp).flatten()[0])
        except Exception:
            # If get_logprob fails, try manual computation for Dirichlet
            try:
                if hasattr(policy, "body") and hasattr(policy, "alpha_layer"):
                    net_out = policy.body(state_jax)
                    alpha_logits = policy.alpha_layer(net_out)
                    alpha = jax.nn.softplus(alpha_logits)

                    from distrax import Dirichlet

                    pi_distribution = Dirichlet(concentration=alpha)
                    clipped_action = jnp.clip(
                        action_batch,
                        getattr(policy, "epsilon", 1e-3),
                        1.0 - getattr(policy, "epsilon", 1e-3),
                    )
                    lp = pi_distribution.log_prob(clipped_action)
                    lp_val = float(jnp.asarray(lp).flatten()[0])
                else:
                    lp_val = 0.0
            except Exception:
                lp_val = 0.0

        log_probs.append(lp_val)

    log_probs = np.array(log_probs)
    # For log10 PDF, convert log probabilities to log10 scale
    log10_pdf_values = log_probs * np.log10(np.e)

    # Create the ternary plot with equilateral triangle aspect ratio
    fig, ax = plt.subplots(
        figsize=(8, 8 * np.sqrt(3) / 2)
    )  # Height adjusted for equilateral triangle

    # Remove figure border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set up the ternary plot without boundary
    figure, tax = ternary.figure(ax=ax, scale=num_points)
    # Remove the boundary line
    # tax.boundary(linewidth=2.0)  # Commented out to remove ternary border

    # Create color map
    cmap = plt.get_cmap("plasma")

    # Create data dict for heatmap - use integer coordinates directly
    data = {}
    for point, pdf_val in zip(points, log10_pdf_values, strict=True):
        # Keep as integers (no normalization back)
        coord = tuple(point.astype(int))
        data[coord] = float(pdf_val)

    # Plot heatmap with proper style and colorbar
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=True)

    # Remove the mean action marker (meaningless for Dirichlet policy)
    # actual_action, _ = policy(state_jax, deterministic=True, rngs=rngs)
    # actual_action_normalized = np.array(actual_action[0]) * num_points
    # tax.scatter([tuple(actual_action_normalized)], marker='*', s=200, c='red', edgecolors='white', linewidths=2, zorder=10, label='Mean Action')

    # Add corner labels with adjusted positioning to avoid title overlap
    tax.right_corner_label("Red", fontsize=12)
    tax.top_corner_label("White", fontsize=12)
    tax.left_corner_label("Blue", fontsize=12)

    # Adjust title with more padding to avoid label overlap
    tax.set_title(f"{policy_name} - Action log10 PDF", fontsize=14, pad=40)
    # Remove legend since we removed the mean action
    # tax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def simulate_rollout(
    actor_critic,
    start_state,
    horizon,
    rngs,
    deterministic=True,
    area_noise_std=0.01,
    clean_area_min=0.0,
    clean_area_max=1.0,
    avg_area_change=0.0,
):
    """
    Simulate a rollout from a start state using the policy.

    State structure (5 dims):
    - state[0]: normalized clean_area
    - state[1:4]: action trace (smoothed actions)

    The action trace can be computed as an exponential moving average:
    trace_t = alpha * action_t + (1 - alpha) * trace_{t-1}

    The dynamics model is simplified:
    - 6 of the 7 state dimensions (action and trace) can be directly calculated from actions
    - The area dimension is estimated with noise around the average change from the dataset

    Args:
        actor_critic: The trained model
        start_state: Initial state (5-dim)
        horizon: Number of steps to simulate
        rngs: Random number generator
        deterministic: Whether to use deterministic policy
        area_noise_std: Standard deviation of noise for area prediction
        clean_area_min: Min value for denormalization
        clean_area_max: Max value for denormalization
        avg_area_change: Average relative change in area from the dataset

    Returns:
        states: Array of shape (horizon+1, 7)
        actions: Array of shape (horizon, 3)
        rewards: Array of shape (horizon,)
    """
    states = [start_state]
    actions = []
    rewards = []

    # Initialize the trace EMA with the initial trace
    trace_ema = UnbiasedExponentialMovingAverage(shape=(3,), alpha=0.5)
    trace_ema.reset()
    trace_ema.update(jnp.array(start_state[4:7]))  # Initialize with initial trace

    current_state = start_state.copy()

    for _ in range(horizon):
        # Get action from policy
        state_jax = jnp.array(current_state).reshape(1, -1)
        action, _ = actor_critic.pi(state_jax, deterministic=deterministic, rngs=rngs)
        action = np.array(action[0])  # Convert to numpy and remove batch dim

        actions.append(action)

        # Estimate reward (proportional change in area with noise)
        # This is a simplified model - real dynamics are more complex
        current_area_norm = current_state[0]
        current_area = (
            current_area_norm * (clean_area_max - clean_area_min) + clean_area_min
        )

        # Simple heuristic: assume some growth based on action
        # In reality this would be learned or more sophisticated
        # For now, add change based on dataset average with noise
        area_change = np.random.normal(avg_area_change, area_noise_std)
        new_area = current_area * (1.0 + area_change)
        new_area = np.clip(new_area, clean_area_min, clean_area_max)
        new_area_norm = (new_area - clean_area_min) / (clean_area_max - clean_area_min)

        # Compute reward as relative change
        reward = (new_area - current_area) / current_area if current_area > 0 else 0.0
        rewards.append(reward)

        # Update action trace using UnbiasedExponentialMovingAverage
        trace_ema.update(jnp.array(action))
        new_trace = trace_ema.compute()

        # Create next state
        next_state = np.zeros(5, dtype=np.float32)
        next_state[0] = new_area_norm  # Updated area
        next_state[1:4] = new_trace  # Updated trace

        states.append(next_state)
        current_state = next_state

    return np.array(states), np.array(actions), np.array(rewards)


def plot_imagined_rollouts(
    actor_critic,
    start_states,
    horizon,
    rngs,
    clean_area_min,
    clean_area_max,
    avg_area_change=0.0,
    save_path=None,
):
    """
    Plot multiple imagined rollouts (both deterministic and stochastic).

    Args:
        actor_critic: The trained model
        start_states: Array of start states (N, 7)
        horizon: Number of steps per rollout
        rngs: Random number generator
        clean_area_min: Min value for denormalization
        clean_area_max: Max value for denormalization
        avg_area_change: Average relative change in area from the dataset
        save_path: Path to save figure
    """
    num_rollouts = len(start_states)

    # Create figure with subplots
    fig, axes = plt.subplots(num_rollouts, 2, figsize=(16, 4 * num_rollouts))
    if num_rollouts == 1:
        axes = axes.reshape(1, -1)

    for i, start_state in enumerate(start_states):
        # Deterministic rollout
        det_states, det_actions, det_rewards = simulate_rollout(
            actor_critic,
            start_state,
            horizon,
            rngs,
            deterministic=True,
            clean_area_min=clean_area_min,
            clean_area_max=clean_area_max,
            avg_area_change=avg_area_change,
        )

        # Stochastic rollout
        stoch_states, stoch_actions, stoch_rewards = simulate_rollout(
            actor_critic,
            start_state,
            horizon,
            rngs,
            deterministic=False,
            area_noise_std=0.02,
            clean_area_min=clean_area_min,
            clean_area_max=clean_area_max,
            avg_area_change=avg_area_change,
        )

        # Plot deterministic rollout
        ax_det = axes[i, 0]
        timesteps = np.arange(horizon)

        ax_det.plot(
            timesteps, det_actions[:, 0], "r-", label="Red", linewidth=2, alpha=0.8
        )
        ax_det.plot(
            timesteps,
            det_actions[:, 1],
            color="gray",
            linestyle="-",
            label="White",
            linewidth=2,
            alpha=0.8,
        )
        ax_det.plot(
            timesteps, det_actions[:, 2], "b-", label="Blue", linewidth=2, alpha=0.8
        )

        ax_det.set_title(
            f"Rollout {i + 1} - Deterministic Policy", fontsize=12, fontweight="bold"
        )
        ax_det.set_xlabel("Time Step")
        ax_det.set_ylabel("Action Coefficient")
        ax_det.legend(loc="upper right")
        ax_det.grid(True, alpha=0.3)
        ax_det.set_ylim(-0.05, 1.05)

        # Add text showing initial area
        init_area_norm = start_state[0]
        init_area = init_area_norm * (clean_area_max - clean_area_min) + clean_area_min
        ax_det.text(
            0.02,
            0.98,
            f"Init Area: {init_area:.2f}",
            transform=ax_det.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot stochastic rollout
        ax_stoch = axes[i, 1]

        ax_stoch.plot(
            timesteps, stoch_actions[:, 0], "r-", label="Red", linewidth=2, alpha=0.8
        )
        ax_stoch.plot(
            timesteps,
            stoch_actions[:, 1],
            color="gray",
            linestyle="-",
            label="White",
            linewidth=2,
            alpha=0.8,
        )
        ax_stoch.plot(
            timesteps, stoch_actions[:, 2], "b-", label="Blue", linewidth=2, alpha=0.8
        )

        ax_stoch.set_title(
            f"Rollout {i + 1} - Stochastic Policy", fontsize=12, fontweight="bold"
        )
        ax_stoch.set_xlabel("Time Step")
        ax_stoch.set_ylabel("Action Coefficient")
        ax_stoch.legend(loc="upper right")
        ax_stoch.grid(True, alpha=0.3)
        ax_stoch.set_ylim(-0.05, 1.05)

        # Add text showing initial area and cumulative reward
        det_cum_reward = det_rewards.sum()
        stoch_cum_reward = stoch_rewards.sum()
        ax_stoch.text(
            0.02,
            0.98,
            f"Init Area: {init_area:.2f}",
            transform=ax_stoch.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax_det.text(
            0.02,
            0.88,
            f"Return: {det_cum_reward:.4f}",
            transform=ax_det.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )
        ax_stoch.text(
            0.02,
            0.88,
            f"Return: {stoch_cum_reward:.4f}",
            transform=ax_stoch.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved imagined rollouts plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize simplex policy on dataset trajectory"
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="results/offline/S8/P0/InAC_GPsim0/0",
        help="Path to experiment directory with trained model",
    )
    parser.add_argument("--dataset", default="plant-rl/continuous-v10", type=str)
    parser.add_argument(
        "--episodes", default=5, type=int, help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--policy_type",
        default="dirichlet",
        type=str,
        choices=["normal", "dirichlet", "mixture_dirichlet", "logistic_normal"],
    )
    parser.add_argument("--state_dim", default=7, type=int)
    parser.add_argument("--action_dim", default=3, type=int)
    parser.add_argument("--hidden_units", default=256, type=int)
    parser.add_argument(
        "--tau",
        default=0.01,
        type=float,
        help="Temperature parameter for clipped advantage",
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Lower clipping threshold for advantage"
    )
    parser.add_argument(
        "--exp_threshold",
        default=10000,
        type=float,
        help="Upper clipping threshold for advantage",
    )
    parser.add_argument(
        "--max_transitions",
        default=10000,
        type=int,
        help="Maximum number of transitions to use for dataset-level advantage computation",
    )
    parser.add_argument(
        "--num_imagined_rollouts",
        default=10,
        type=int,
        help="Number of imagined rollouts to generate",
    )
    parser.add_argument(
        "--rollout_horizon",
        default=14,
        type=int,
        help="Horizon for imagined rollouts",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.exp_path) / "plots"
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = minari.load_dataset(args.dataset)

    episodes = list(dataset.iterate_episodes())
    episode_indexes = np.linspace(0, len(episodes) - 1, args.episodes, dtype=int)

    # Compute average area change from dataset
    all_rewards = []
    for episode in episodes:
        all_rewards.extend(episode.rewards)
    avg_area_change = float(np.mean(all_rewards))
    print(f"Average relative area change (reward) in dataset: {avg_area_change:.6f}")

    # Load model
    exp_path = Path(args.exp_path)
    print(f"Loading model from {exp_path}")
    actor_critic: ActorCritic = load_model(
        exp_path=exp_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_units=args.hidden_units,
        policy_type=args.policy_type,
    )  # type: ignore

    # Get actions
    rngs = nnx.Rngs(42)  # Fixed seed for reproducibility
    for episode_idx in episode_indexes:
        print(f"\n{'=' * 60}\nProcessing episode index: {episode_idx}\n{'=' * 60}")
        args.episode_idx = episode_idx

        episode = episodes[args.episode_idx]
        print(
            f"Selected episode {args.episode_idx} with {len(episode.observations) - 1} steps"
        )
        states, dataset_actions, policy_actions, beh_policy_actions = (
            get_trajectory_actions(episode, actor_critic.pi, actor_critic.beh_pi, rngs)
        )

        # Get rewards from episode
        rewards = episode.rewards

        print(f"Dataset actions shape: {dataset_actions.shape}")
        print(f"Policy actions shape: {policy_actions.shape}")
        print(f"Behavior policy actions shape: {beh_policy_actions.shape}")
        print(
            f"Dataset actions sum check: {dataset_actions.sum(axis=1).mean():.3f} ± {dataset_actions.sum(axis=1).std():.3f}"
        )
        print(
            f"Policy actions sum check: {policy_actions.sum(axis=1).mean():.3f} ± {policy_actions.sum(axis=1).std():.3f}"
        )
        print(
            f"Beh policy actions sum check: {beh_policy_actions.sum(axis=1).mean():.3f} ± {beh_policy_actions.sum(axis=1).std():.3f}"
        )

        # Plot trajectory
        trajectory_plot_path = (
            output_dir / f"simplex_policy_trajectory_ep{args.episode_idx}.png"
        )
        plot_trajectory_actions(
            states,
            dataset_actions,
            policy_actions,
            beh_policy_actions,
            args.episode_idx,
            trajectory_plot_path,
        )

        # Plot comprehensive episode figure
        comprehensive_plot_path = (
            output_dir / f"episode_comprehensive_ep{args.episode_idx}.png"
        )
        plot_episode_comprehensive(
            actor_critic,
            states,
            dataset_actions,
            policy_actions,
            beh_policy_actions,
            args.episode_idx,
            tau=args.tau,
            eps=args.eps,
            exp_threshold=args.exp_threshold,
            save_path=comprehensive_plot_path,
        )

        # Plot comprehensive episode figure with ternary plots
        comprehensive_ternary_plot_path = (
            output_dir / f"episode_comprehensive_ternary_ep{args.episode_idx}.png"
        )
        plot_episode_comprehensive_ternary(
            actor_critic,
            states,
            dataset_actions,
            policy_actions,
            beh_policy_actions,
            args.episode_idx,
            tau=args.tau,
            eps=args.eps,
            exp_threshold=args.exp_threshold,
            num_points=32,
            save_path=comprehensive_ternary_plot_path,
        )

        # Plot ternary timeseries (grid of ternary plots over time)
        ternary_timeseries_plot_path = (
            output_dir / f"episode_ternary_timeseries_ep{args.episode_idx}.png"
        )
        plot_episode_ternary_timeseries(
            actor_critic,
            states,
            dataset_actions,
            rewards,
            args.episode_idx,
            rngs,
            tau=args.tau,
            eps=args.eps,
            exp_threshold=args.exp_threshold,
            num_points=24,
            num_timesteps=8,
            save_path=ternary_timeseries_plot_path,
        )

        # print("=" * 60)

        # # Plot Q-values ternary for each timestep
        # print(f"\nGenerating ternary plots for {len(states)} timesteps...")
        # for t, state in enumerate(states):
        #     print(f"  Timestep {t}/{len(states) - 1}...")

        #     # Q-values
        #     ternary_plot_path = (
        #         output_dir / f"simplex_q_ternary_ep{args.episode_idx}_t{t:03d}.png"
        #     )
        #     plot_q_ternary(actor_critic, state, save_path=ternary_plot_path)
        #     print("    ✓ Q-values")

        #     # Advantage
        #     advantage_plot_path = (
        #         output_dir
        #         / f"simplex_advantage_ternary_ep{args.episode_idx}_t{t:03d}.png"
        #     )
        #     plot_advantage_ternary(actor_critic, state, save_path=advantage_plot_path)
        #     print("    ✓ Advantage")

        #     # Value
        #     value_plot_path = (
        #         output_dir / f"simplex_value_ternary_ep{args.episode_idx}_t{t:03d}.png"
        #     )
        #     plot_value_ternary(actor_critic, state, save_path=value_plot_path)
        #     print("    ✓ Value")

        #     # Clipped Advantage
        #     clipped_adv_plot_path = (
        #         output_dir
        #         / f"simplex_clipped_adv_ternary_ep{args.episode_idx}_t{t:03d}.png"
        #     )
        #     plot_clipped_advantage_ternary(
        #         actor_critic,
        #         state,
        #         tau=args.tau,
        #         eps=args.eps,
        #         exp_threshold=args.exp_threshold,
        #         save_path=clipped_adv_plot_path,
        #     )
        #     print("    ✓ Clipped Advantage")

        #     # Policy (pi) distribution
        #     pi_plot_path = (
        #         output_dir / f"simplex_pi_ternary_ep{args.episode_idx}_t{t:03d}.png"
        #     )
        #     plot_policy_ternary(
        #         actor_critic.pi, state, rngs, policy_name="pi", save_path=pi_plot_path
        #     )
        #     print("    ✓ Policy (pi)")

        #     # Behavior policy (beh_pi) distribution
        #     beh_pi_plot_path = (
        #         output_dir / f"simplex_beh_pi_ternary_ep{args.episode_idx}_t{t:03d}.png"
        #     )
        #     plot_policy_ternary(
        #         actor_critic.beh_pi,
        #         state,
        #         rngs,
        #         policy_name="beh_pi",
        #         save_path=beh_pi_plot_path,
        #     )
        #     print("    ✓ Behavior policy (beh_pi)")

    # Generate imagined rollouts
    print("\n" + "=" * 60)
    print(f"Generating {args.num_imagined_rollouts} imagined rollouts...")
    print("=" * 60)

    # Estimate the min/max from the normalized values
    # Since the data is already normalized in [0, 1], we assume it represents
    # the full range of clean_area values in the original data
    clean_area_min = 0.0  # Normalized minimum
    clean_area_max = 1.0  # Normalized maximum

    # Select start states from different episodes
    start_state_indices = np.linspace(
        0, len(episodes) - 1, args.num_imagined_rollouts, dtype=int
    )
    start_states = []

    for idx in start_state_indices:
        episode = episodes[idx]
        # Take the first state of the episode as a start state
        start_state = episode.observations[0]
        start_states.append(start_state)
        print(f"  Selected start state from episode {idx}, area={start_state[0]:.3f}")

    start_states = np.array(start_states)

    # Generate and plot rollouts
    rollouts_plot_path = output_dir / "imagined_rollouts.png"
    plot_imagined_rollouts(
        actor_critic,
        start_states,
        horizon=args.rollout_horizon,
        rngs=rngs,
        clean_area_min=clean_area_min,
        clean_area_max=clean_area_max,
        avg_area_change=avg_area_change,
        save_path=rollouts_plot_path,
    )

    print(f"Imagined rollouts plot saved to {rollouts_plot_path}")

    # # Compute and plot dataset-level metrics (advantage, value, action-value, clipped advantage)
    # print("\n" + "=" * 60)
    # print("Computing mean metrics over entire dataset...")
    # print("=" * 60)
    # dataset_metrics = compute_dataset_metrics(
    #     actor_critic,
    #     dataset,
    #     tau=args.tau,
    #     eps=args.eps,
    #     exp_threshold=args.exp_threshold,
    #     num_points=64,
    #     max_transitions=args.max_transitions,
    # )

    # # Plot advantage
    # advantage_plot_path = output_dir / "simplex_advantage_dataset_mean.png"
    # plot_dataset_metric_ternary(
    #     dataset_metrics,
    #     "advantages",
    #     "Mean Advantage over Dataset",
    #     num_points=64,
    #     cmap_name="RdBu_r",
    #     save_path=advantage_plot_path,
    # )

    # # Plot value
    # value_plot_path = output_dir / "simplex_value_dataset_mean.png"
    # plot_dataset_metric_ternary(
    #     dataset_metrics,
    #     "values",
    #     "Mean Value over Dataset",
    #     num_points=64,
    #     cmap_name="viridis",
    #     save_path=value_plot_path,
    # )

    # # Plot action-value (Q)
    # action_value_plot_path = output_dir / "simplex_action_value_dataset_mean.png"
    # plot_dataset_metric_ternary(
    #     dataset_metrics,
    #     "action_values",
    #     "Mean Action-Value (Q) over Dataset",
    #     num_points=64,
    #     cmap_name="viridis",
    #     save_path=action_value_plot_path,
    # )

    # # Plot clipped advantage
    # clipped_adv_plot_path = output_dir / "simplex_clipped_adv_dataset_mean.png"
    # plot_dataset_metric_ternary(
    #     dataset_metrics,
    #     "clipped_advantages",
    #     "Mean Clipped Advantage over Dataset",
    #     num_points=64,
    #     cmap_name="inferno",
    #     save_path=clipped_adv_plot_path,
    # )
    # print("=" * 60)

    # # Compute and plot dataset-level policy metrics
    # print("\n" + "=" * 60)
    # print("Computing mean policy log probs over entire dataset...")
    # print("=" * 60)

    # # Policy (pi)
    # dataset_pi_metrics = compute_dataset_policy_metrics(
    #     actor_critic.pi,
    #     dataset,
    #     num_points=64,
    #     max_transitions=args.max_transitions,
    # )
    # pi_dataset_plot_path = output_dir / "simplex_pi_dataset_mean.png"
    # plot_dataset_policy_ternary(
    #     dataset_pi_metrics,
    #     policy_name="Policy (pi)",
    #     num_points=64,
    #     save_path=pi_dataset_plot_path,
    # )

    # # Behavior policy (beh_pi)
    # dataset_beh_pi_metrics = compute_dataset_policy_metrics(
    #     actor_critic.beh_pi,
    #     dataset,
    #     num_points=64,
    #     max_transitions=args.max_transitions,
    # )
    # beh_pi_dataset_plot_path = output_dir / "simplex_beh_pi_dataset_mean.png"
    # plot_dataset_policy_ternary(
    #     dataset_beh_pi_metrics,
    #     policy_name="Behavior Policy (beh_pi)",
    #     num_points=64,
    #     save_path=beh_pi_dataset_plot_path,
    # )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
