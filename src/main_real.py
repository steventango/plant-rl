import asyncio
import os
import shutil
import sys

sys.path.append(os.getcwd())
import argparse
import logging
import socket
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity, Ignore, MovingAverage, Subsample
from PyExpUtils.collection.utils import Pipe
from PyExpUtils.results.sqlite import saveCollector

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import log
from utils.preempt import TimeoutHandler
from utils.RlGlue.rl_glue import AsyncRLGlue

default_save_keys = {"left", "right"}

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

logging.basicConfig(
    level=logging.ERROR,
    format='[%(asctime)s] %(levelname)s:%(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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


def save_images(env, data_path: Path, save_keys):
    isoformat = env.time.isoformat(timespec='seconds').replace(':', '')
    zone_identifier = env.zone.identifier
    images_path = data_path / f"z{zone_identifier}" / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    for key, image in env.images.items():
        if save_keys != "*" and key not in save_keys:
            continue
        img_path = images_path / f"{isoformat}_{key}.jpg"
        image = image.convert("RGB")
        image.save(img_path, "JPEG", quality=90)


def backup_and_save(exp, collector, idx, base):
    context = exp.buildSaveContext(idx, base=base)
    db_file = context.resolve('results.db')
    db_file_bak = context.resolve('results.db.bak')
    if os.path.exists(db_file):
        shutil.move(db_file, db_file_bak)
    saveCollector(exp, collector, base=base)

async def main():
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
                    "state": Identity(),
                    "action": Identity(),
                    "reward": Identity(),
                    "steps": Identity(),
                    "time": Identity(),
                    "area": Identity(),
                },
                # by default, ignore keys that are not explicitly listed above
                default=Ignore(),
            ),
        )
        collector.setIdx(idx)
        run = exp.getRun(idx)

        # set random seeds accordingly
        np.random.seed(run)

        # build stateful things and attach to checkpoint
        problem = chk.build('p', lambda: Problem(exp, idx, collector))
        agent = chk.build('a', problem.getAgent)
        env = chk.build('e', problem.getEnvironment)

        glue = chk.build("glue", lambda: AsyncRLGlue(agent, env))
        chk.initial_value('episode', 0)

        context = exp.buildSaveContext(idx, base=args.save_path)
        agent_path = Path(context.resolve()).relative_to('results')
        data_path = Path('/data') / agent_path
        images_save_keys = problem.exp_params.get("image_save_keys", default_save_keys)
        (data_path / f"z{env.zone.identifier}").mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)

        config = {
            **problem.params,
            "context": str(agent_path)
        }

        wandb_run = wandb.init(
            entity="plant-rl",
            project="main",
            notes=str(agent_path),
            config=config,
            settings=wandb.Settings(
                x_stats_disk_paths=("/", "/data"),
            ),
        )

        # Run the experiment
        start_time = time.time()
        # Track the current active save task (if any)
        current_save_task = None

        # if we haven't started yet, then make the first interaction
        if glue.total_steps == 0:
            s, a, info = await glue.start()
            episode = chk['episode']
            log(env, glue, wandb_run, s, a, info, episode=episode)
            save_images(env, data_path, images_save_keys)

        for step in range(glue.total_steps, exp.total_steps):
            collector.next_frame()
            if problem.exp_params.get("checkpoint", True):
                # Cancel the previous save task if it's still running and not completed
                if current_save_task and not current_save_task.done():
                    current_save_task.cancel()
                    logger.debug(f"Cancelled previous checkpoint save at step {step}")

                # Create a new task to run chk.save concurrently
                current_save_task = asyncio.create_task(asyncio.to_thread(chk.save))

            interaction = await glue.step()
            collector.collect('time', env.time.timestamp())
            collector.collect('state', interaction.o)
            collector.collect('action', interaction.a)
            collector.collect('reward', interaction.r)
            collector.collect('steps', glue.num_steps)
            # for key, value in interaction.extra.items():
            #     collector.collect(key, value.astype(np.float64))
            episodic_return = glue.total_reward if interaction.t else None
            episode = chk['episode']
            log(env, glue, wandb_run, interaction.o, interaction.a, interaction.extra, interaction.r, interaction.t, episodic_return, episode)

            save_images(env, data_path, images_save_keys)

            if interaction.t or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
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
                logger.debug(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')

                s, a, info = await glue.start()
                log(env, glue, wandb_run, s, a, info)
                save_images(env, data_path, images_save_keys)

            backup_and_save(exp, collector, idx, args.save_path)

        collector.reset()

        env.close()

        # Wait for all checkpoint save tasks to complete before exiting
        if current_save_task and not current_save_task.done():
            logger.debug("Waiting for the last checkpoint save task to complete...")
            await current_save_task
            logger.debug("Last checkpoint save task completed.")

        # ------------
        # -- Saving --
        # ------------
        backup_and_save(exp, collector, idx, args.save_path)
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main())
