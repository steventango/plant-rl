#!/usr/bin/env python3
"""
Generate a poster figure with three simplexes:
1. RGB gradient visualization
2. Q-values for a specific state
3. Learned policy for the same state
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import minari
import numpy as np
import optax
import seaborn as sns
import ternary
from flax import nnx

from algorithms.nn.inac.agent.base import load
from algorithms.nn.inac.agent.in_sample import ActorCritic, Optimizers

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

    # Use load to restore parameters
    actor_critic, optimizers = load(
        nnx.split(actor_critic)[0],
        nnx.split(optimizers)[0],
        parameters_dir,
    )

    return actor_critic


def generate_simplex_points(num_points=64):
    """Generate all points on a simplex grid."""
    points = []
    for i in range(num_points + 1):
        for j in range(num_points + 1 - i):
            k = num_points - i - j
            points.append([i, j, k])

    points = np.array(points, dtype=float)
    simplex_actions = points / num_points
    return points, simplex_actions


def plot_rgb_gradient_simplex(ax, num_points=64):
    """
    Plot a simplex with RGB gradient colors.
    Red at right corner, White at top corner, Blue at left corner.
    Creates a proper color blend: pink between white-red, purple between red-blue, etc.
    """
    # Remove spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Create ternary plot
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Generate points and compute RGB values
    points, simplex_actions = generate_simplex_points(num_points)

    # Create RGB color for each point by blending the three base colors
    # simplex_actions[:, 0] = Red coefficient (maps to pure red: [1, 0, 0])
    # simplex_actions[:, 1] = White coefficient (maps to white: [1, 1, 1])
    # simplex_actions[:, 2] = Blue coefficient (maps to pure blue: [0, 0, 1])

    rgb_colors = []
    for action in simplex_actions:
        # Blend: Red=[1,0,0], White=[1,1,1], Blue=[0,0,1]
        r = action[0] * 1.0 + action[1] * 1.0 + action[2] * 0.0
        g = action[0] * 0.0 + action[1] * 1.0 + action[2] * 0.0
        b = action[0] * 0.0 + action[1] * 1.0 + action[2] * 1.0

        rgb = np.array([r, g, b])
        # Normalize to ensure we stay in [0, 1] range
        rgb = np.clip(rgb, 0, 1)
        rgb_colors.append(rgb)

    rgb_colors = np.array(rgb_colors)

    # We'll use a custom approach: create a ListedColormap from our RGB colors
    # and plot with heatmap using indices
    from matplotlib.colors import ListedColormap

    data = {}
    color_dict = {}
    for i, point in enumerate(points):
        coord = tuple(point.astype(int))
        data[coord] = float(i)
        color_dict[coord] = rgb_colors[i]

    # Create a custom colormap with exactly the colors we need
    cmap = ListedColormap(rgb_colors)

    # Plot heatmap with our custom colormap
    # Set vmin and vmax to match our data range
    tax.heatmap(
        data,
        style="hexagonal",
        cmap=cmap,
        colorbar=False,
        vmin=0,
        vmax=len(points) - 1,
    )

    return tax


def plot_q_values_simplex(ax, actor_critic, state, num_points=64):
    """
    Plot Q-values over the simplex for a given state.
    """
    # Remove spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Create ternary plot
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Generate simplex points
    points, simplex_actions = generate_simplex_points(num_points)

    # Convert to JAX arrays
    actions_jax = jnp.array(simplex_actions)
    state_jax = jnp.array(state).reshape(1, -1)

    # Compute Q-values in batch
    @nnx.jit
    def compute_q_values(actor_critic, state, actions):
        state_tiled = jnp.tile(state, (actions.shape[0], 1))
        return actor_critic.q(state_tiled, actions)

    q_values, _, _ = compute_q_values(actor_critic, state_jax, actions_jax)
    q_values_np = np.array(q_values)

    # Create data dict
    data = {}
    for point, q_val in zip(points, q_values_np, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(q_val)

    # Plot heatmap
    cmap = plt.get_cmap("viridis")
    tax.heatmap(data, style="hexagonal", cmap=cmap, colorbar=False, cbarlabel="Q-value")

    # Get vertex values
    vertices = np.array([[num_points, 0, 0], [0, num_points, 0], [0, 0, num_points]])
    vertex_q = []
    for v in vertices:
        coord = tuple(v.astype(int))
        vertex_q.append(data[coord])

    return tax


def plot_policy_simplex(ax, policy, state, rngs, num_points=64):
    """
    Plot policy probability density over the simplex for a given state.
    """
    # Remove spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Create ternary plot
    figure, tax = ternary.figure(ax=ax, scale=num_points)

    # Generate simplex points
    points, simplex_actions = generate_simplex_points(num_points)

    # Convert to JAX arrays
    actions_jax = jnp.array(simplex_actions)
    state_jax = jnp.array(state).reshape(1, -1)

    # Compute log probabilities in batch
    @nnx.jit
    def compute_log_probs(policy, state, actions):
        state_tiled = jnp.tile(state, (actions.shape[0], 1))
        return policy.get_logprob(state_tiled, actions)

    try:
        log_probs_jax = compute_log_probs(policy, state_jax, actions_jax)
        log_probs = np.array(log_probs_jax).flatten()
    except Exception:
        log_probs = np.full(actions_jax.shape[0], -np.inf)

    log_probs = np.array(log_probs)
    # Convert to log10 scale for better visualization
    log10_pdf_values = log_probs * np.log10(np.e)

    # Create data dict
    data = {}
    for point, pdf_val in zip(points, log10_pdf_values, strict=True):
        coord = tuple(point.astype(int))
        data[coord] = float(pdf_val)

    # Plot heatmap
    cmap = plt.get_cmap("viridis")
    tax.heatmap(
        data, style="hexagonal", cmap=cmap, colorbar=False, cbarlabel="log₁₀ π(a|s)"
    )

    return tax


def generate_poster_figure(
    actor_critic,
    state,
    rngs,
    num_points=64,
    save_path=None,
    dpi=300,
):
    """
    Generate a poster figure with three simplexes in a row.

    Args:
        actor_critic: Trained actor-critic model
        state: State vector to visualize
        rngs: Random number generators
        num_points: Resolution of simplex grid
        save_path: Path to save the figure
        dpi: Resolution of saved figure
    """
    # Create figure with 3 subplots in a row
    fig, axs = plt.subplots(1, 3, figsize=(6.66, 1.89), layout="constrained")

    # Plot 1: RGB Gradient
    print("Generating RGB gradient simplex...")
    plot_rgb_gradient_simplex(axs[0], num_points=num_points)

    # Plot 2: Q-values
    print("Generating Q-values simplex...")
    plot_q_values_simplex(axs[1], actor_critic, state, num_points=num_points)

    # Plot 3: Policy
    print("Generating policy simplex...")
    plot_policy_simplex(axs[2], actor_critic.pi, state, rngs, num_points=num_points)

    # Add overall title
    # fig.suptitle(
    #     "Plant RL: Action Space Visualization",
    #     fontsize=20,
    #     fontweight="bold",
    #     y=0.98,
    # )

    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"Saved poster figure to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate poster figure with three simplexes"
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="src/algorithms/nn/inac/data/JAX/output/plant-rl/continuous-v8/0/0_run",
        help="Path to experiment directory with trained model",
    )
    parser.add_argument(
        "--dataset",
        default="plant-rl/continuous-v8",
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--episode",
        default=0,
        type=int,
        help="Episode index to visualize",
    )
    parser.add_argument(
        "--timestep",
        default=0,
        type=int,
        help="Timestep within episode to visualize",
    )
    parser.add_argument(
        "--policy_type",
        default="mixture_dirichlet",
        type=str,
        choices=["normal", "dirichlet", "mixture_dirichlet", "logistic_normal"],
        help="Type of policy",
    )
    parser.add_argument(
        "--state_dim",
        default=7,
        type=int,
        help="State dimension",
    )
    parser.add_argument(
        "--action_dim",
        default=3,
        type=int,
        help="Action dimension",
    )
    parser.add_argument(
        "--hidden_units",
        default=256,
        type=int,
        help="Number of hidden units",
    )
    parser.add_argument(
        "--num_points",
        default=600,
        type=int,
        help="Resolution of simplex grid",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the figure (default: auto-generate in exp_path/plots)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="DPI for saved figure",
    )

    args = parser.parse_args()

    # Set up output path
    if args.output is None:
        output_dir = Path(args.exp_path) / "plots"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = (
            output_dir / f"poster_figure_ep{args.episode}_t{args.timestep}.png"
        )
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = minari.load_dataset(args.dataset)

    # Get the specified episode and timestep
    episodes = list(dataset.iterate_episodes())
    if args.episode >= len(episodes):
        raise ValueError(
            f"Episode {args.episode} not found. Dataset has {len(episodes)} episodes."
        )

    episode = episodes[args.episode]
    states = episode.observations[:-1]  # Remove terminal state

    if args.timestep >= len(states):
        raise ValueError(
            f"Timestep {args.timestep} not found. Episode has {len(states)} timesteps."
        )

    state = states[args.timestep]
    print(
        f"Using episode {args.episode}, timestep {args.timestep} (total: {len(states)} timesteps)"
    )
    print(f"State: {state}")

    # Load model
    exp_path = Path(args.exp_path)
    print(f"Loading model from {exp_path}")
    actor_critic = load_model(
        exp_path=exp_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_units=args.hidden_units,
        policy_type=args.policy_type,
    )

    # Create RNG
    rngs = nnx.Rngs(42)

    # Generate poster figure
    print("\nGenerating poster figure...")
    generate_poster_figure(
        actor_critic=actor_critic,
        state=state,
        rngs=rngs,
        num_points=args.num_points,
        save_path=output_path,
        dpi=args.dpi,
    )

    print(f"\n✓ Done! Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
