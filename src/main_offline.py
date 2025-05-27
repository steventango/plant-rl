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

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import expand
from utils.plotting import get_Q, plot_q, plot_q_diff  # Added import
from utils.preempt import TimeoutHandler
from utils.RlGlue.rl_glue import LoggingRlGlue
from utils.window_avg import WindowAverage

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
    if glue.total_steps == 0:
        s, env_info = env.start()
        agent.load_start(s, env_info)
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
        data_exhausted = env_info.get("exhausted", False)
        if term:
            agent.load_end(reward, env_info)
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()
            if not data_exhausted:
                s, env_info = env.start()
                agent.load_start(s, env_info)
                data['observation'].append(s)
                data['action'].append(None)
                data['reward'].append(None)
                data['terminal'].append(None)
                data_exhausted = env_info.get("exhausted", False)
        else:
            agent.load_step(reward, s, env_info)

    df = pd.DataFrame(data)
    df.to_csv(context.resolve('data.csv'), index=False)

    for step in range(exp.total_steps):
        info = agent.plan()
        if step % 1000 == 0:
            expanded_info = {}
            for key, value in info.items():
                expanded_info.update(expand(key, value))
            wandb_run.log(expanded_info, step=step)

        if step == 0 or step % 10 ** int(np.log10(step)) == 0:
            # Plot and save Q-values
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
                    daytime_observation_space = np.linspace(0, 1, 12 * 6, endpoint=True)
                    area_observation_space = np.linspace(0, 1, 100, endpoint=True)

                    num_actions = weights.shape[0]
                    Q_vals = get_Q(weights, tile_coder, daytime_observation_space, area_observation_space, num_actions)

                    # Plot and save Q-values
                    q_plot_filename = q_plots_dir / f"q_values_step_{step:06d}.jpg"
                    plot_q(daytime_observation_space, area_observation_space, Q_vals)  # Call the plot function
                    plt.savefig(q_plot_filename)  # Save the figure
                    plt.close()  # Close the figure

                    # Plot and save Q-value differences
                    if num_actions >= 2:
                        Q_diff = Q_vals[:, :, 1] - Q_vals[:, :, 0]
                    else:
                        logger.info(
                            f"Step {step}: Not enough actions ({num_actions}) to compute Q-difference. Plotting Q[s,0] instead."
                        )
                        Q_diff = Q_vals[:, :, 0]

                    q_diff_plot_filename = q_plots_dir / f"q_diff_step_{step:06d}.jpg"
                    plot_q_diff(daytime_observation_space, area_observation_space, Q_diff)  # Call the plot function
                    plt.savefig(q_diff_plot_filename)  # Save the figure
                    plt.close()  # Close the figure
                    logger.info(f"Step {step}: Saved Q-value plots to {q_plots_dir}")
                except Exception as e:
                    logger.error(f"Step {step}: Error during Q-value plotting: {e}")

            else:
                logger.warning(
                    f"Step {step}: Agent does not have 'w' or 'tile_coder' attributes, or they are None. Skipping Q-value plotting."
                )

    chk.save()

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
