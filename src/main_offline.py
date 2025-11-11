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
import minari
import numpy as np
import torch
from tqdm import tqdm

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler

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

    # Load offline dataset
    dataset_name = exp_params["dataset"]
    logger.info(f"Loading offline dataset: {dataset_name}")
    dataset = minari.load_dataset(dataset_name)

    # Build problem to get environment/agent configuration
    problem = chk.build("p", lambda: Problem(exp, idx, None))

    # Build agent using problem's getAgent method
    agent = chk.build("a", problem.getAgent)

    # Load offline data into agent's buffer if agent supports it
    logger.info("Loading offline dataset into agent buffer...")
    agent.load(dataset)

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
    max_steps = exp_params.get("max_steps", exp.total_steps)
    log_interval = exp_params.get("log_interval", 10000)
    eval_interval = exp_params.get("eval_interval", 100000)

    logger.info(f"Starting offline training for {max_steps} steps")

    # Create plots directory
    plots_dir = exp_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_time = time.time()
    total_steps = 0
    losses_accumulator = defaultdict(list)

    # Create progress bar
    pbar = tqdm(
        total=max_steps,
        desc="Training",
        unit="step",
        disable=prod,
        dynamic_ncols=True,
    )

    while total_steps < max_steps:
        if (
            eval_interval
            and total_steps % eval_interval == 0
            and hasattr(agent, "actor_critic")
        ):
            logger.info(f"Evaluating at step {total_steps}")
            pbar.set_description("Evaluating")
            try:
                from algorithms.nn.inac.agent.base import evaluate_on_dataset

                evaluate_on_dataset(
                    logger,
                    total_steps,
                    dataset,
                    agent.actor_critic.pi,
                    agent.actor_critic.q,
                    agent.rngs,
                    plots_dir=plots_dir,
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
            pbar.set_description("Training")

        # Perform update step using the plan() method (RL-Glue interface)
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
                postfix_dict = {k: f"{v:.3f}" for k, v in avg_losses.items()}
                postfix_dict["steps/s"] = f"{elapsed_time:.1f}"
                pbar.set_postfix(postfix_dict)

                logger.info(
                    f"TRAIN LOG: steps {total_steps}, "
                    f"{total_steps * 100 // max_steps}%, "
                    f"{elapsed_time:.2f} steps/s"
                )

                # Log individual losses
                loss_str = "\nLOSSES:\n" + "\n".join(
                    f"{k} {v:.3f}" for k, v in avg_losses.items()
                )
                logger.info(loss_str)

                # Log to wandb
                wandb_log = {
                    **{f"{k}_loss": v for k, v in avg_losses.items()},
                    "steps_per_second": elapsed_time,
                }
                wandb_run.log(wandb_log, step=total_steps)
            else:
                # Just log progress if no losses
                logger.info(
                    f"TRAIN LOG: steps {total_steps}, "
                    f"{total_steps * 100 // max_steps}%, "
                    f"{elapsed_time:.2f} steps/s"
                )
                wandb_run.log({"steps_per_second": elapsed_time}, step=total_steps)

            # Reset accumulators and timer
            losses_accumulator = defaultdict(list)
            start_time = time.time()

    pbar.close()
    logger.info("Offline training complete")

    # Save final model (InAC-specific)
    if hasattr(agent, "actor_critic") and hasattr(agent, "optimizers"):
        try:
            from algorithms.nn.inac.agent.base import save

            save_path = exp_path / idx / "parameters"
            save_path.mkdir(parents=True, exist_ok=True)
            save(agent.actor_critic, agent.optimizers, save_path)  # type: ignore
            logger.info(f"Saved final model to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save model parameters: {e}")

    # Save checkpoint
    chk.save()
    logger.info("Checkpoint saved")

    wandb_run.finish()
