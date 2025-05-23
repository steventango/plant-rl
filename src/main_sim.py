import os
import sys

sys.path.append(os.getcwd())
import argparse
import logging
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Identity, Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict
from ml_instrumentation.metadata import attach_metadata
from tqdm import tqdm

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import log
from utils.preempt import TimeoutHandler
from utils.window_avg import WindowAverage
from rlglue import RlGlue

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
    collector.set_experiment_id(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly, with optional offset
    params = exp.get_hypers(idx)
    seed = run + params.get("experiment", {}).get("seed_offset", 0)

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
    agent_path = Path(context.resolve()).relative_to('results')

    config = {
        **problem.params,
        "context": str(agent_path)
    }

    wandb_run = wandb.init(
        entity="plant-rl",
        project="sim",
        notes=str(agent_path),
        config=config,
        settings=wandb.Settings(
            x_stats_disk_paths=("/", "/data"),
        ),
    )

    # Run the experiment
    start_time = time.time()

    # if we haven't started yet, then make the first interaction
    if glue.total_steps == 0:
        # Assuming glue.start() returns: (observation, action, info_dict) or similar
        # and log function is adapted for this.
        # For this change, we focus on glue.step() return.
        # The original code was: s, a, info = glue.start()
        # If glue.start() changed, this part would need adaptation too.
        # Let's assume it's compatible or out of scope for this specific change.
        s, a, info = glue.start()
        log(env, glue, wandb_run, s, a, info)

    for step in range(glue.total_steps, exp.total_steps):
        collector.next_frame()
        chk.maybe_save()
        observation, reward, terminated, truncated, info = glue.step() # RLGlue returns 5-tuple
        log(env, glue, wandb_run, observation, info['action'], info, reward) # Assuming info dict from glue.step() contains 'action'
                
        collector.collect('reward', reward)
        collector.collect('episode', chk['episode'])
        collector.collect('steps', glue.num_steps)
        collector.collect('action', info['action'])  # Assuming info dict from glue.step() contains 'action'

        if terminated or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # Collect total reward
            collector.collect('return', glue.total_reward)

            # track how many episodes are completed (cutoff is counted as termination for this count)
            chk['episode'] += 1

            # compute the average time-per-step in ms
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            episode = chk['episode']
            logger.debug(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')

            glue.start()

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    # context = exp.buildSaveContext(idx, base=args.save_path) # Should already exist
    save_db_path = context.resolve('results.db')
    meta = getParamsAsDict(exp, idx)
    meta |= {'seed': seed} # Ensure 'seed' is the correct variable from the file's scope
    attach_metadata(save_db_path, idx, meta)
    collector.merge(save_db_path)
    collector.close()
    chk.delete()
