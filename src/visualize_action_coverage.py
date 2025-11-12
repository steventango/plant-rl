#!/usr/bin/env python3
"""
Visualize action space coverage from the dataset using ternary plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import minari
import numpy as np
import seaborn as sns
import ternary

sns.set_style("white")


def collect_all_actions(dataset):
    """Collect all actions from all episodes in the dataset."""
    all_actions = []

    for episode in dataset.iterate_episodes():
        actions = episode.actions
        all_actions.append(actions)

    return np.vstack(all_actions)


def plot_action_coverage_ternary(actions, num_bins=50, save_path=None):
    """
    Create a ternary plot showing the density/coverage of actions over the simplex.

    Args:
        actions: Array of shape (N, 3) with actions on the simplex
        num_bins: Number of bins for discretizing the simplex
        save_path: Path to save the plot
    """
    # Verify actions are on simplex
    action_sums = actions.sum(axis=1)
    print(f"Action sum check: {action_sums.mean():.6f} ± {action_sums.std():.6f}")
    print(f"Action sum range: [{action_sums.min():.6f}, {action_sums.max():.6f}]")

    # Normalize actions to sum to 1 (handle numerical errors)
    actions_normalized = actions / action_sums[:, np.newaxis]

    # Verify normalization
    normalized_sums = actions_normalized.sum(axis=1)
    print(
        f"Normalized sum check: {normalized_sums.mean():.6f} ± {normalized_sums.std():.6f}"
    )

    # Create bins for counting - use finer grid for better density visualization
    scale = num_bins

    # Generate ALL possible points on the simplex grid
    all_points = []
    for i in range(scale + 1):
        for j in range(scale + 1 - i):
            k = scale - i - j
            all_points.append([i, j, k])
    all_points = np.array(all_points)

    print(f"Total possible bins: {len(all_points)}")

    # For each grid point, compute density using KDE-like approach
    # Convert grid points to normalized coordinates
    grid_normalized = all_points / scale

    # Compute distances from each grid point to all actions
    # Use a simple counting approach with tolerance
    tolerance = 1.5 / scale  # Slightly larger than bin size

    densities = {}
    for grid_point, grid_norm in zip(all_points, grid_normalized, strict=True):
        # Count actions within tolerance
        distances = np.linalg.norm(actions_normalized - grid_norm, axis=1)
        count = np.sum(distances < tolerance)
        if count > 0:
            densities[tuple(grid_point)] = count

    print(f"Grid points with data: {len(densities)}")

    if len(densities) == 0:
        print("WARNING: No density data computed. Using direct binning instead.")
        # Fallback to direct binning
        action_coords = (actions_normalized * scale).astype(int)
        action_coords = np.clip(action_coords, 0, scale)

        for coord in action_coords:
            coord_sum = coord.sum()
            if coord_sum != scale:
                diff = scale - coord_sum
                coord[coord.argmax()] += diff
            key = tuple(coord)
            densities[key] = densities.get(key, 0) + 1

    # Create the ternary plot
    fig, ax = plt.subplots(figsize=(10, 9))

    # Set up the ternary plot
    figure, tax = ternary.figure(ax=ax, scale=scale)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=scale // 10, color="gray", alpha=0.3)

    # Create color map
    max_density = max(densities.values())
    min_density = min(densities.values())
    print(f"Density range: [{min_density}, {max_density}]")

    # Use log scale for better visualization
    vmin, vmax = 0, np.log10(max_density + 1)
    log_densities = {k: np.log10(v + 1) for k, v in densities.items()}

    tax.heatmap(
        log_densities,
        style="hexagonal",
        cmap="YlOrRd",
        colorbar=True,
        vmin=vmin,
        vmax=vmax,
        cbarlabel="log10(density + 1)",
    )
    print("Using log scale for density visualization")

    # Add corner labels
    tax.right_corner_label("Red", fontsize=14, offset=0.18)
    tax.top_corner_label("White", fontsize=14, offset=0.18)
    tax.left_corner_label("Blue", fontsize=14, offset=0.18)

    tax.set_title("Dataset Action Coverage over Simplex", fontsize=16, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved coverage plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_action_marginals(actions, save_path=None):
    """Plot marginal distributions for each action component."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ["red", "gray", "blue"]
    labels = ["Red", "White", "Blue"]

    for i, (ax, color, label) in enumerate(zip(axes, colors, labels, strict=True)):
        ax.hist(actions[:, i], bins=50, color=color, alpha=0.7, edgecolor="black")
        ax.set_xlabel(f"{label} Coefficient", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{label} Distribution", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = actions[:, i].mean()
        ax.axvline(
            mean_val,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"μ={mean_val:.3f}",
        )
        ax.legend()

    plt.suptitle("Action Component Marginal Distributions", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved marginals plot to {save_path}")
    else:
        plt.show()


def plot_action_scatter_3d(actions, save_path=None):
    """Plot 3D scatter of actions."""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Sample if too many points
    if len(actions) > 5000:
        indices = np.random.choice(len(actions), 5000, replace=False)
        plot_actions = actions[indices]
    else:
        plot_actions = actions

    scatter = ax.scatter(
        plot_actions[:, 0],
        plot_actions[:, 1],
        plot_actions[:, 2],
        c=plot_actions[:, 1],
        cmap="viridis",
        alpha=0.5,
        s=10,
    )

    ax.set_xlabel("Red", fontsize=12)
    ax.set_ylabel("White", fontsize=12)
    ax.set_zlabel("Blue", fontsize=12)
    ax.set_title("3D Action Distribution", fontsize=16)

    # Draw simplex edges
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                "k-",
                linewidth=2,
                alpha=0.5,
            )

    plt.colorbar(scatter, ax=ax, label="White coefficient")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D scatter plot to {save_path}")
    else:
        plt.show()


def compute_statistics(actions):
    """Compute and print statistics about action coverage."""
    print("\n" + "=" * 60)
    print("ACTION COVERAGE STATISTICS")
    print("=" * 60)

    print(f"\nTotal actions: {len(actions)}")

    print("\nComponent statistics:")
    labels = ["Red", "White", "Blue"]
    for i, label in enumerate(labels):
        print(f"  {label}:")
        print(f"    Mean: {actions[:, i].mean():.4f}")
        print(f"    Std:  {actions[:, i].std():.4f}")
        print(f"    Min:  {actions[:, i].min():.4f}")
        print(f"    Max:  {actions[:, i].max():.4f}")

    # Check for corner concentration (pure strategies)
    corner_threshold = 0.9
    corners = [
        (actions[:, 0] > corner_threshold, "Red-dominant"),
        (actions[:, 1] > corner_threshold, "White-dominant"),
        (actions[:, 2] > corner_threshold, "Blue-dominant"),
    ]

    print(f"\nCorner concentration (>{corner_threshold}):")
    for mask, name in corners:
        count = mask.sum()
        percentage = count / len(actions) * 100
        print(f"  {name}: {count} ({percentage:.2f}%)")

    # Check for edge concentration (two components)
    edge_threshold = 0.05
    edges = [
        ((actions[:, 0] < edge_threshold), "Red-Blue edge"),
        ((actions[:, 1] < edge_threshold), "Red-White edge"),
        ((actions[:, 2] < edge_threshold), "White-Blue edge"),
    ]

    print(f"\nEdge concentration (component <{edge_threshold}):")
    for mask, name in edges:
        count = mask.sum()
        percentage = count / len(actions) * 100
        print(f"  {name}: {count} ({percentage:.2f}%)")

    # Center concentration
    center_threshold = 0.15
    center_mask = (np.abs(actions - 1 / 3) < center_threshold).all(axis=1)
    center_count = center_mask.sum()
    print(f"\nCenter concentration (all components ~1/3 ± {center_threshold}):")
    print(f"  {center_count} ({center_count / len(actions) * 100:.2f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize action space coverage from dataset"
    )
    parser.add_argument(
        "--dataset",
        default="plant-rl/continuous-v8",
        type=str,
        help="Minari dataset name",
    )
    parser.add_argument(
        "--output_dir", default="plots", type=str, help="Directory to save plots"
    )
    parser.add_argument(
        "--num_bins",
        default=64,
        type=int,
        help="Number of bins for ternary plot discretization",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = minari.load_dataset(args.dataset)

    # Collect all actions
    print("Collecting actions from all episodes...")
    actions = collect_all_actions(dataset)
    print(f"Collected {len(actions)} actions")

    # Compute statistics
    compute_statistics(actions)

    # Create plots
    print("\nGenerating plots...")

    # Ternary coverage plot
    coverage_path = output_dir / "action_coverage_ternary.png"
    plot_action_coverage_ternary(
        actions, num_bins=args.num_bins, save_path=coverage_path
    )

    # Marginal distributions
    marginals_path = output_dir / "action_marginals.png"
    plot_action_marginals(actions, save_path=marginals_path)

    # 3D scatter
    scatter_3d_path = output_dir / "action_scatter_3d.png"
    plot_action_scatter_3d(actions, save_path=scatter_3d_path)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
