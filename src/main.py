import Box2D     # we need to import this first because cedar is stupid
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm

import time
import socket
import logging
import argparse
import numpy as np
from RlGlue import RlGlue
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from problems.registry import getProblem
from PyExpUtils.results.sqlite import saveCollector
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore, MovingAverage, Subsample, Identity
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

logfile = '/Users/oliverdiamond/Desktop/alberta/research/plant-rl/src/exp.log'
logging.basicConfig(filename=logfile, level=logging.ERROR)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
if not prod:
    logger.setLevel(logging.DEBUG)
    
# Separate terminal logger
term_logger = logging.getLogger('exp_term')
term_logger.setLevel(logging.DEBUG if not prod else logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))  # Simple format for terminal
term_logger.addHandler(console_handler)

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
            'return': Identity(),
            'episode': Identity(),
            'steps': Identity(),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Ignore(),
    ))
    collector.setIdx(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly, with optional offset
    params = exp.get_hypers(idx)
    seed = run + params.get("experiment", {}).get("seed_offset", 0)
    np.random.seed(seed)

    # build stateful things and attach to checkpoint
    problem = chk.build('p', lambda: Problem(exp, idx, collector))
    agent = chk.build('a', problem.getAgent)
    env = chk.build('e', problem.getEnvironment)

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
        #if glue.total_steps == 1 or glue.total_steps == 1000 or glue.total_steps == 5000 or glue.total_steps == 10000 or glue.total_steps == 15000:
            #term_logger.info(f'\n\n{glue.agent.policy_str}')


        if interaction.t or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # collect some data
            collector.collect('return', glue.total_reward)
            collector.collect('episode', chk['episode'])
            collector.collect('steps', glue.num_steps)

            # track how many episodes are completed (cutoff is counted as termination for this count)
            chk['episode'] += 1

            # compute the average time-per-step in ms
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            episode = chk['episode']
            #term_logger.info(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')
            term_logger.info(f'{glue.num_steps} {glue.total_reward}')
            glue.start()
        #logger.info(glue.agent.info)
    #term_logger.info(f'\n{glue.agent.policy_str}')

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
    chk.delete()
