import os
import sys

sys.path.append(os.getcwd())
import argparse
import logging
import random
import socket
import time
from collections import defaultdict
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import minari
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
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

for idx in indices:
    chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
    loaded = chk.load_if_exists()
    if loaded:
        logger.info("Loaded checkpoint")
    timeout_handler.before_cancel(chk.save)

    run = exp.getRun(idx)

    # Get hyperparameters from experiment
    params = exp.get_hypers(idx)
    exp_params = params.get("experiment", {})
    agent_params = params.get("agent", {})

    # Set random seeds accordingly, with optional offset
    seed = run + exp_params.get("seed_offset", 0)

    # Seed various modules
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Build context for saving results
    context = exp.buildSaveContext(idx, base=args.save_path)
    Path(context.resolve()).mkdir(parents=True, exist_ok=True)
    agent_path = Path(context.resolve()).relative_to("results")
    exp_path = Path(context.resolve())

    # Build problem to get environment/agent configuration
    problem = chk.build("p", lambda: Problem(exp, idx, None))

    # Build agent using problem's getAgent method
    agent = chk.build("a", problem.getAgent)

    online_training = params.get("update_per_step", 0) > 0

    env = chk.build("e", problem.getEnvironment)
    eval_env = chk.build("ee", problem.getEvalEnvironment)

    dataset_id = params.get("dataset_id")
    if dataset_id:
        logger.info(f"Loading offline dataset: {dataset_id}")
        dataset = minari.load_dataset(dataset_id)
        agent.load(dataset)

    def eval_rollouts(n_rollouts, rollout_steps):
        returns = []

        all_rollouts_actions = []

        for i in range(n_rollouts):
            obs, info = eval_env.start()
            current_return = 0.0
            rollout_actions = []
            # Create directory for this rollout's images if it's one of the first 10
            if i < 10:
                rollout_dir = plots_dir / "trajectories" / f"rollout_{i}"
                rollout_dir.mkdir(parents=True, exist_ok=True)
                # Save image if available and we are tracing this rollout
                try:
                    img_array = eval_env.render()
                    if img_array is not None:
                        img = Image.fromarray(img_array)
                        img.save(rollout_dir / "step_0.png")
                except Exception as e:
                    logger.warning(f"Failed to save initial image: {e}")

            for t in range(1, rollout_steps + 1):
                obs_jax = jax.numpy.array([obs])  # Add batch dim
                action, _ = agent.actor_critic.pi(
                    obs_jax, deterministic=True, rngs=agent.rngs
                )

                # Convert action back to numpy/list for env
                # Action from agent is [batch, action_dim]
                action_vec = np.array(action)[0]
                rollout_actions.append(action_vec)

                reward, next_obs, done, info = eval_env.step(action_vec)
                obs = next_obs
                current_return += reward

                # Save image if available and we are tracing this rollout
                if i < 10:
                    try:
                        img_array = eval_env.render()
                        if img_array is not None:
                            img = Image.fromarray(img_array)
                            img.save(rollout_dir / f"step_{t}.png")
                    except Exception as e:
                        logger.warning(f"Failed to save image at step {t}: {e}")

                if done:
                    break

            returns.append(current_return)
            all_rollouts_actions.append(rollout_actions)
        return returns, all_rollouts_actions

    def plot_actions(all_rollouts_actions):
        # Handle variable lengths by padding with NaNs
        max_len = max(len(r) for r in all_rollouts_actions)
        if max_len > 0 and len(all_rollouts_actions) > 0:
            action_dim = len(all_rollouts_actions[0][0])
            action_tensor = np.full((n_rollouts, max_len, action_dim), np.nan)

            for r_idx, r_actions in enumerate(all_rollouts_actions):
                for step_idx, step_action in enumerate(r_actions):
                    action_tensor[r_idx, step_idx, :] = step_action

            # Compute mean ignoring NaNs
            mean_actions = np.nanmean(
                action_tensor, axis=0
            )  # Shape: (max_len, action_dim)

            fig, ax = plt.subplots(figsize=(6, 4))

            # Stackplot with specific colors
            steps = range(mean_actions.shape[0])
            # Transpose for stackplot (needs y as (M, N))
            y_data = mean_actions.T
            colors = ["red", "grey", "blue"]

            ax.stackplot(steps, y_data, colors=colors, alpha=0.7)

            ax.set_xlabel("Step")
            ax.set_ylabel("Average Action Value (Stacked)")
            ax.set_title(f"Average Actions over {n_rollouts} Rollouts (Stacked)")
            # despine
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            wandb.log(
                {"eval/mean_actions": wandb.Image(fig)},
                step=total_steps,
            )
            plt.close(fig)

    def plot_returns(returns):
        # violin plot of returns
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=returns, ax=ax)
        sns.despine(ax=ax)
        returns_plot_path = plots_dir / "calibration_returns.png"
        plt.savefig(returns_plot_path)
        wandb.log(
            {"eval/returns_dist": wandb.Image(str(returns_plot_path))},
            step=total_steps,
        )
        plt.close(fig)

    # Setup wandb
    config = {
        **problem.params,
        "context": str(agent_path),
    }

    wandb_run = wandb.init(
        entity="plant-rl",
        project="offline",
        notes=str(agent_path),
        config=config,
        settings=wandb.Settings(
            x_stats_disk_paths=("/", "/data"),
        ),
    )

    # Training configuration
    log_interval = exp_params.get("log_interval", 10000)
    eval_interval = exp_params.get("eval_interval", 100000)

    logger.info(f"Starting offline training for {exp.total_steps} steps")

    # Create plots directory
    plots_dir = exp_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_time = time.time()
    total_steps = 0
    losses_accumulator = defaultdict(list)

    glue = chk.build("glue", lambda: LoggingRlGlue(agent, env))

    # Create progress bar
    pbar = tqdm(
        total=exp.total_steps,
        desc="Training",
        unit="step",
        disable=prod,
        dynamic_ncols=True,
    )

    while total_steps < exp.total_steps:
        if (
            eval_interval
            and total_steps % eval_interval == 0
            and hasattr(agent, "actor_critic")
        ):
            # ---------------------------
            # -- Calibration Rollouts --
            # ---------------------------
            if hasattr(agent, "actor_critic"):
                n_rollouts = 64
                rollout_steps = 13
                returns, all_rollouts_actions = eval_rollouts(
                    n_rollouts=n_rollouts, rollout_steps=rollout_steps
                )
                mean_return = np.mean(returns)
                wandb.log({"eval/return": mean_return}, step=total_steps)

                # --- Plots ---
                plot_actions(all_rollouts_actions)
                plot_returns(returns)

        if online_training:
            interaction = glue.step()
            info = interaction.extra

            if interaction.t:
                glue.start()
        else:
            # Perform update step using the plan() method
            t0 = time.time()
            info = agent.plan()

        # Accumulate losses if info is returned as a dict
        if isinstance(info, dict):
            for key, value in info.items():
                if isinstance(value, (int, float, np.number)):
                    losses_accumulator[key].append(float(value))

        total_steps += 1
        pbar.update(1)

        # Log at intervals
        if log_interval and total_steps % log_interval == 0:
            elapsed_time = log_interval / (time.time() - start_time)

            if losses_accumulator:
                avg_losses = {k: np.mean(v) for k, v in losses_accumulator.items()}

                # Update progress bar with loss information
                postfix_dict = {k: f"{v:.2f}" for k, v in avg_losses.items()}
                postfix_dict["eval/return"] = f"{mean_return:.2f}"
                pbar.set_postfix(postfix_dict)

                # Log to wandb
                wandb_log = {
                    **{f"{k}_loss": v for k, v in avg_losses.items()},
                    "steps_per_second": elapsed_time,
                }
                wandb_run.log(wandb_log, step=total_steps)
            else:
                wandb_run.log({"steps_per_second": elapsed_time}, step=total_steps)

            # Reset accumulators and timer
            losses_accumulator = defaultdict(list)
            start_time = time.time()

    pbar.close()
    logger.info("Offline training complete")

    # Save final model (InAC-specific)
    if hasattr(agent, "actor_critic") and hasattr(agent, "optimizers"):
        from algorithms.nn.inac.agent.base import save

        save_path = exp_path / str(idx) / "parameters"
        save_path.mkdir(parents=True, exist_ok=True)
        save(agent.actor_critic, agent.optimizers, save_path)  # type: ignore
        logger.info(f"Saved final model to {save_path}")

    wandb_run.finish()
