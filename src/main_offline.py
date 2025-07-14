import os
import sys
from collections import defaultdict

import pandas as pd

from utils.plotting import (
    plot_q_values_and_diff,
    plot_state_action_distribution,
    plot_trajectories,
)

sys.path.append(os.getcwd())
import argparse
import logging
import random
import socket
import time
from pathlib import Path

import jax
import numpy as np
import torch
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity, Ignore
from PyExpUtils.results.sqlite import saveCollector

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import expand
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

    chk.save()

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
    # -- Saving --
    # ------------
