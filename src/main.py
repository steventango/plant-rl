import Box2D     # we need to import this first because cedar is stupid
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm

import time
import random
import socket
import logging
import argparse
import numpy as np
import torch
from RlGlue import RlGlue
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from problems.registry import getProblem
from PyExpUtils.results.sqlite import saveCollector
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore, MovingAverage, Subsample, Identity
from utils.window_avg import WindowAverage
from PyExpUtils.collection.utils import Pipe

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
            'action_is_optimal': Identity(),
        },
        default=Ignore(),
    ))
    collector.setIdx(idx)
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

    # If exp.total_steps is -1, then set total steps such that each run lasts for "environment.last_day" days.
    if (exp.problem == 'PlantSimulator' or exp.problem == 'SimplePlantSimulator') and exp.total_steps == -1:
        problem.params['total_steps'] = env.terminal_step
        exp.total_steps = env.terminal_step

    glue = chk.build('glue', lambda: RlGlue(agent, env))
    chk.initial_value('episode', 0)

    # Run the experiment
    start_time = time.time()

    # if we haven't started yet, then make the first interaction
    if glue.total_steps == 0:
        glue.start()

    for step in range(glue.total_steps, exp.total_steps):
        collector.next_frame()
        chk.maybe_save()
        interaction = glue.step()

        collector.collect('reward', interaction.r)
        collector.collect('episode', chk['episode'])
        collector.collect('steps', glue.num_steps)
        collector.collect('action', interaction.a)      # or int.from_bytes(glue.last_action, byteorder='little') for GAC
        collector.collect('action_is_optimal', interaction.extra.get('action_is_optimal', -1))   # Check if the agent took the optimal action

        if interaction.t or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
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
    saveCollector(exp, collector, base=args.save_path)
    chk.delete()
