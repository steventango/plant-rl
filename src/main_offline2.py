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

import matplotlib.pyplot as plt  # Added import
import numpy as np
import torch
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity, Ignore, MovingAverage, Subsample
from PyExpUtils.collection.utils import Pipe
from PyExpUtils.results.sqlite import saveCollector
from tqdm import tqdm
import seaborn as sns
from utils.logger import expand
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
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
import jax

device = 'gpu' if args.gpu else 'cpu'
jax.config.update('jax_platform_name', device)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
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

    collector = chk.build('collector', lambda: Collector(
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'return': Identity(),       # total reward at the end of episode
            'reward': Identity(),       # reward at each step
            'episode': Identity(),
            'steps': Identity(),
            'action': Identity(),
        },
        default=Ignore(),
    ))
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
    problem = chk.build('p', lambda: Problem(exp, idx, collector))
    agent = chk.build('a', problem.getAgent)
    env = chk.build('e', problem.getEnvironment)

    glue = chk.build('glue', lambda: LoggingRlGlue(agent, env))
    chk.initial_value('episode', 0)

    context = exp.buildSaveContext(idx, base=args.save_path)
    Path(context.resolve()).mkdir(parents=True, exist_ok=True)
    agent_path = Path(context.resolve()).relative_to('results')

    # Create directory for Q-value plots
    q_plots_dir = Path(context.resolve()) / "q_value_plots"
    q_plots_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **problem.params,
        "context": str(agent_path)
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

    # Run the experiment
    start_time = time.time()

    data = defaultdict(list)

    # if we haven't started yet, then make the first interaction
    data_exhausted = False
    step = 0
    if glue.total_steps == 0:
        s, env_info = env.start()
        agent.start(s, env_info)
        data_exhausted = env_info.get("exhausted", False)
        data['observation'].append(s)
        data['action'].append(None)
        data['reward'].append(None)
        data['terminal'].append(None)

    while not data_exhausted:
        (reward, s, term, env_info) = env.step(None)
        data['observation'].append(s)
        data['action'].append(env_info.get("action", None))
        data['reward'].append(reward)
        data['terminal'].append(term)
        if reward != 0:
            logger.info(f"Step {step}: Received reward {reward} at state {s}")
        data_exhausted = env_info.get("exhausted", False)
        if term:
            # log the reward
            logger.info(f"Step {step}: Episode finished with reward {reward}")
            agent.end(reward, env_info)
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()
            if not data_exhausted:
                s, env_info = env.start()
                agent.start(s, env_info)
                data['observation'].append(s)
                data['action'].append(None)
                data['reward'].append(None)
                data['terminal'].append(None)
                data_exhausted = env_info.get("exhausted", False)
            # Plot and save Q-values
            if (
                hasattr(agent, "w")
                and agent.w is not None
            ):
                try:
                    weights = agent.w

                    num_actions = weights.shape[0]
                    n = 13
                    Q_vals = np.eye(13) @ weights.T

                    # plot and save Q-values
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10)) # Create 2 subplots
                    ax1 = axes[0]
                    ax2 = axes[1]

                    # Plot heatmap of Q-values on the first subplot
                    sns.heatmap(
                        Q_vals.T,
                        ax=ax1,
                        cmap="viridis",
                        cbar=True,
                        cbar_kws={"label": "Q-value"},
                        xticklabels=np.arange(n),
                        yticklabels=np.arange(num_actions),
                    )
                    ax1.set_title(f"Q-values at step {step}")
                    ax1.set_xlabel("Daytime Observation Space")
                    ax1.set_ylabel("Actions")
                    ax1.set_xticks(np.arange(n))
                    ax1.set_xticklabels([f"{t + 9}" for t in np.arange(n)])
                    ax1.set_yticks(np.arange(num_actions))
                    ax1.set_yticklabels([f"Action {i}" for i in range(num_actions)])
                    ax1.invert_yaxis()  # Invert y-axis to have the first action on top

                    # Calculate and plot the difference Q_vals[:, 1] - Q_vals[:, 0] on the second subplot
                    if num_actions >= 2:
                        q_diff = Q_vals[:, 1] - Q_vals[:, 0]
                        ax2.plot(np.arange(n), q_diff, marker='o', linestyle='-')
                        ax2.set_title(f"Q-value Difference (Action 1 - Action 0) at step {step}")
                        ax2.set_xlabel("Daytime Observation Space")
                        ax2.set_ylabel("Q-value Difference")
                        ax2.set_xticks(np.arange(n))
                        ax2.set_xticklabels([f"{t + 9}" for t in np.arange(n)])
                        ax2.grid(True)
                    else:
                        ax2.text(0.5, 0.5, "Not enough actions to compute difference.",
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=ax2.transAxes)
                        ax2.set_title(f"Q-value Difference at step {step}")


                    plt.tight_layout()
                    # Save the Q-value plot
                    q_plot_filename = q_plots_dir / f"q_values_step_{step:06d}.png"
                    plt.savefig(q_plot_filename)  # Save the figure
                    plt.close(fig)  # Close the figure
                    logger.info(f"Step {step}: Saved Q-value plots to {q_plots_dir}")
                except Exception as e:
                    logger.error(f"Step {step}: Error during Q-value plotting: {e}")

            else:
                logger.warning(
                    f"Step {step}: Agent does not have 'w' or 'tile_coder' attributes, or they are None. Skipping Q-value plotting."
                )
        else:
            _, info = agent.step(reward, s, env_info)
            expanded_info = {}
            for key, value in info.items():
                expanded_info.update(expand(key, value))
            wandb_run.log(expanded_info, step=step)

        step += 1

    df = pd.DataFrame(data)
    df.to_csv(context.resolve('data.csv'), index=False)


    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
