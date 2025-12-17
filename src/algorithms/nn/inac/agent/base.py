import os
from collections import Counter
from pathlib import Path

import flashbax as fbx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import pandas as pd
import seaborn as sns
from flax import nnx
from gymnasium.spaces import Discrete
from minari import MinariDataset


def fill_offline_data_to_buffer(dataset: MinariDataset, batch_size: int):
    dataset_size = dataset.total_steps
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_terminations = []
    all_truncations = []

    for episode in dataset.iterate_episodes():
        all_obs.append(episode.observations[:-1])
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)
        all_next_obs.append(episode.observations[1:])
        all_terminations.append(episode.terminations)
        all_truncations.append(episode.truncations)

    truncations = jnp.concatenate(all_truncations, axis=0)
    valid_mask = ~truncations

    dataset_transitions = {
        "state": jnp.concatenate(all_obs, axis=0)[valid_mask],
        "action": jnp.concatenate(all_actions, axis=0)[valid_mask],
        "reward": jnp.concatenate(all_rewards, axis=0)[valid_mask],
        "next_state": jnp.concatenate(all_next_obs, axis=0)[valid_mask],
        "termination": jnp.concatenate(all_terminations, axis=0)[valid_mask],
    }

    dummy_transition = jax.tree_util.tree_map(lambda x: x[0], dataset_transitions)
    replay = fbx.make_flat_buffer(
        max_length=dataset_size,
        min_length=batch_size,
        sample_batch_size=batch_size,
    )
    replay_state = replay.init(dummy_transition)
    add_fn = jax.jit(replay.add, donate_argnums=(0,))

    def add_transition(carry, transition):
        replay_state = carry
        replay_state = add_fn(replay_state, transition)
        return replay_state, None

    replay_state, _ = jax.lax.scan(add_transition, replay_state, dataset_transitions)
    return replay, replay_state


def evaluate_on_dataset(
    logger,
    total_steps: int,
    dataset: MinariDataset,
    pi: nnx.Module,
    q: nnx.Module,
    rngs: nnx.Rngs,
    plots_dir: Path,
):
    """
    Evaluate policy and critic on offline dataset by analyzing behavior at different area bins.

    Args:
        logger: Logger instance
        total_steps: Current training step
        dataset: MinariDataset to evaluate on
        pi: Policy network
        q: Q-network
        rngs: Random number generators
        plots_dir: Directory to save plots
    """
    # Collect states from each area bin across all episodes
    num_area_bins = 8
    bin_states = [[] for _ in range(num_area_bins)]
    bin_actions = [[] for _ in range(num_area_bins)]
    bin_min_q_dataset = [None] * num_area_bins
    bin_min_q_policy = [None] * num_area_bins

    for episode in dataset.iterate_episodes():
        episode_length = len(episode.observations) - 1
        for t in range(episode_length):
            observation = episode.observations[t]
            actual_area = observation[1] if len(observation) > 1 else 0.0
            # Bin by hundreds: 0-100, 100-200, 200-300, 300-400, etc.
            area_bin_idx = int(actual_area // 100)
            area_bin_idx = min(
                max(area_bin_idx, 0), num_area_bins - 1
            )  # Clamp to valid range

            bin_states[area_bin_idx].append(observation)
            bin_actions[area_bin_idx].append(episode.actions[t])

    is_discrete = isinstance(dataset.action_space, Discrete)
    num_actions = dataset.action_space.n if is_discrete else None
    bin_q_per_action = (
        [[[] for _ in range(num_actions)] for _ in range(num_area_bins)]
        if is_discrete and num_actions is not None
        else None
    )

    @nnx.jit
    def compute_evaluation(states, actions, pi, q, rngs):
        predicted_actions, _ = pi(states, deterministic=True, rngs=rngs)
        min_q_dataset, q1_dataset, q2_dataset = q(states, actions)
        min_q_policy, q1_policy, q2_policy = q(states, predicted_actions)
        return (
            predicted_actions,
            min_q_dataset,
            q1_dataset,
            q2_dataset,
            min_q_policy,
            q1_policy,
            q2_policy,
        )

    @nnx.jit
    def compute_q_min(states, actions, q):
        min_q, _, _ = q(states, actions)
        return min_q

    logger.info(f"\n{'=' * 60}")
    logger.info(f"DATASET EVALUATION at step {total_steps}")
    logger.info(f"{'=' * 60}")

    for d in range(num_area_bins):
        if not bin_states[d]:
            continue

        states = jnp.array(bin_states[d])
        actions = jnp.array(bin_actions[d])

        # Get policy predictions and Q-values (JIT compiled)
        (
            predicted_actions,
            min_q_dataset,
            q1_dataset,
            q2_dataset,
            min_q_policy,
            q1_policy,
            q2_policy,
        ) = compute_evaluation(states, actions, pi, q, rngs)

        # Store Q-values for plotting
        bin_min_q_dataset[d] = min_q_dataset
        bin_min_q_policy[d] = min_q_policy

        # Calculate statistics
        mean_predicted_action = predicted_actions.mean(axis=0)
        std_predicted_action = predicted_actions.std(axis=0)
        mean_dataset_action = actions.mean(axis=0)

        mean_q_dataset = min_q_dataset.mean()
        mean_q_policy = min_q_policy.mean()

        logger.info(f"Area bin {d}:")
        logger.info(f"  Samples: {len(states)}")

        if is_discrete and num_actions is not None:
            # Squeeze actions if they are in shape (N, 1)
            if actions.ndim > 1 and actions.shape[1] == 1:
                actions = actions.squeeze(axis=1)
            if predicted_actions.ndim > 1 and predicted_actions.shape[1] == 1:
                predicted_actions = predicted_actions.squeeze(axis=1)

            # Compute Q-values for all actions in all states
            q_all = jnp.stack(
                [
                    compute_q_min(
                        states, jnp.full((states.shape[0],), i, dtype=jnp.int32), q
                    )
                    for i in range(num_actions)
                ],
                axis=1,
            )  # (batch, num_actions)
            if bin_q_per_action is not None:
                for i in range(num_actions):
                    bin_q_per_action[d][i] = q_all[:, i].tolist()

            # Dataset action proportions
            unique_dataset, counts_dataset = jnp.unique(
                actions, return_counts=True, size=num_actions
            )
            props_dataset = counts_dataset / len(actions)
            logger.info("  Dataset action proportions:")
            for i, prop in enumerate(props_dataset):
                if counts_dataset[i] > 0:
                    logger.info(f"    Action {unique_dataset[i]}: {prop:.2f}")

            # Policy action proportions
            unique_policy, counts_policy = jnp.unique(
                predicted_actions, return_counts=True, size=num_actions
            )
            props_policy = counts_policy / len(predicted_actions)
            logger.info("  Policy action proportions:")
            for i, prop in enumerate(props_policy):
                if counts_policy[i] > 0:
                    logger.info(f"    Action {unique_policy[i]}: {prop:.2f}")

            # Q-values for each action
            logger.info("  Q-values (dataset actions):")
            for i in range(num_actions):
                if counts_dataset[i] > 0:
                    action_mask = actions == unique_dataset[i]
                    q_vals = q_all[action_mask, i]
                    logger.info(f"    Action {unique_dataset[i]}: {q_vals.mean():.3f}")

        else:  # Continuous actions
            logger.info(f"  Dataset action mean: {mean_dataset_action}")
            logger.info(f"  Policy action mean:  {mean_predicted_action}")
            logger.info(f"  Policy action std:   {std_predicted_action}")
            logger.info(f"  Q-value (dataset actions): {mean_q_dataset:.3f}")
            logger.info(f"  Q-value (policy actions):  {mean_q_policy:.3f}")
            logger.info(
                f"  Q-value difference:        {(min_q_policy - min_q_dataset).mean():.3f}"
            )

            # Log Q-values at simplex vertices
            red_action = jnp.array([1.0, 0.0, 0.0])
            white_action = jnp.array([0.0, 1.0, 0.0])
            blue_action = jnp.array([0.0, 0.0, 1.0])

            red_actions = jnp.tile(red_action, (states.shape[0], 1))
            white_actions = jnp.tile(white_action, (states.shape[0], 1))
            blue_actions = jnp.tile(blue_action, (states.shape[0], 1))

            q_red, _, _ = q(states, red_actions)  # type: ignore
            q_white, _, _ = q(states, white_actions)  # type: ignore
            q_blue, _, _ = q(states, blue_actions)  # type: ignore

            logger.info(f"  Q-value at [1,0,0] (red):    {q_red.mean():.3f}")
            logger.info(f"  Q-value at [0,1,0] (white):  {q_white.mean():.3f}")
            logger.info(f"  Q-value at [0,0,1] (blue):   {q_blue.mean():.3f}")

    logger.info(f"{'=' * 60}\n")

    # Generate plots of Q-values per action for discrete actions
    if is_discrete and num_actions is not None and bin_q_per_action is not None:
        # Bar plots with 95% CI
        fig, axes = plt.subplots(num_area_bins, 1, figsize=(8, 4 * num_area_bins))
        if num_area_bins == 1:
            axes = [axes]
        for d in range(num_area_bins):
            data = []
            for i in range(num_actions):
                if bin_q_per_action[d][i]:  # Check if list is not empty
                    for val in bin_q_per_action[d][i]:
                        data.append({"action": i, "q": float(val)})
            if data:
                df = pd.DataFrame(data)
                sns.barplot(
                    data=df,
                    x="action",
                    y="q",
                    ci=95,
                    capsize=0.1,
                    palette=["red", "white", "blue"],
                    ax=axes[d],
                )
                # Add border to white bar
                for patch in axes[d].patches:
                    if patch.get_facecolor()[:3] == (1.0, 1.0, 1.0):  # white bar
                        patch.set_edgecolor("black")
                        patch.set_linewidth(1)
                axes[d].set_title(
                    f"Mean Q-values with 95% CI at Area bin {d} (n={len(bin_states[d])})"
                )
                axes[d].set_xlabel("Action")
                axes[d].set_ylabel("Q-value")
                actions_array = jnp.array(bin_actions[d])
                if actions_array.ndim > 1 and actions_array.shape[1] == 1:
                    actions_array = actions_array.squeeze(axis=1)
                action_counts = Counter(actions_array.tolist())
                counts = [action_counts.get(i, 0) for i in range(num_actions)]
                labels = [
                    f"{name} (n={count})"
                    for name, count in zip(
                        ["Red", "White", "Blue"], counts, strict=True
                    )
                ]
                axes[d].set_xticklabels(labels)
            else:
                axes[d].set_title(f"No data at Area bin {d}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"q_bar_step_{total_steps}.png"))
        plt.close()

    return None


def save(module: nnx.Module, optimizers: nnx.Module, parameters_dir):
    parameters_dir = os.path.abspath(parameters_dir)
    module_state = nnx.split(module)[1]
    optimizers_state = nnx.split(optimizers)[1]

    ckpt = {
        "module": module_state,
        "optimizers": optimizers_state,
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(os.path.join(parameters_dir, "default"), ckpt, force=True)


def load(
    module: nnx.GraphDef,
    optimizers: nnx.GraphDef,
    parameters_dir,
    module_state=None,
    optimizers_state=None,
):
    parameters_dir = os.path.abspath(parameters_dir)
    target = None
    if module_state is not None and optimizers_state is not None:
        target = {
            "module": module_state,
            "optimizers": optimizers_state,
        }

    with ocp.StandardCheckpointer() as checkpointer:
        # If target is provided, it forces restoration onto the devices of the target
        # ignoring the sharding saved in the checkpoint.
        kwargs = {"target": target} if target is not None else {}
        ckpt = checkpointer.restore(
            os.path.join(parameters_dir, "default"), **kwargs
        )

    module = nnx.merge(module, ckpt["module"])
    optimizers = nnx.merge(optimizers, ckpt["optimizers"])
    return module, optimizers
