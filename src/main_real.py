import asyncio
import os
import shutil
import sys
import warnings
import aiofiles

sys.path.append(os.getcwd())
import argparse
import logging
import socket
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity, Ignore, MovingAverage, Subsample
from PyExpUtils.collection.utils import Pipe
from PyExpUtils.results.sqlite import saveCollector

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import expand, log
from utils.preempt import TimeoutHandler
from utils.RlGlue.rl_glue import AsyncRLGlue, Interaction

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
parser.add_argument('-d', '--deploy', action='store_true', default=False, 
                    help='Allows for easily restarting logging for crashed runs in deployment. If the run already exists, then wandb will resume logging to the same run.')

# ---------------------------
# -- Library Configuration --
# ---------------------------
import jax

# Moved args parsing and jax config to if __name__ == "__main__"

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

# Moved exp, indices, Problem to main function

def save_images(env, dataset_path: Path, save_keys):
    isoformat = env.time.isoformat(timespec='seconds').replace(':', '')
    images_path = dataset_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    img_path = None
    for key, image in env.images.items():
        if save_keys != "*" and key not in save_keys:
            continue
        img_path = images_path / f"{isoformat}_{key}.jpg"
        image = image.convert("RGB")
        image.save(img_path, "JPEG", quality=90)
    return img_path.name if img_path else None


def backup_and_save(exp, collector, idx, base):
    context = exp.buildSaveContext(idx, base=base)
    db_file = context.resolve('results.db')
    db_file_bak = context.resolve('results.db.bak')
    if os.path.exists(db_file):
        shutil.copy(db_file, db_file_bak)
    saveCollector(exp, collector, base=base) # exp will be passed or available to backup_and_save

async def main(args_namespace):
    exp = ExperimentModel.load(args_namespace.exp)
    indices = args_namespace.idxs
    Problem = getProblem(exp.problem)

    for idx in indices:
        chk = Checkpoint(exp, idx, base_path=args_namespace.checkpoint_path)
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
                    "terminal": Identity(),
                    "reward": Identity(),
                    "steps": Identity(),
                    "time": Identity(),
                    "area": Identity(),
                },
                # by default, ignore keys that are not explicitly listed above
                default=Identity(),
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

        context = exp.buildSaveContext(idx, base=args_namespace.save_path)
        agent_path = Path(context.resolve()).relative_to('results')
        dataset_path = Path('/data') / agent_path  / f"z{env.zone.identifier}"
        images_save_keys = problem.exp_params.get("image_save_keys", default_save_keys)
        dataset_path.mkdir(parents=True, exist_ok=True)
        raw_csv_path = dataset_path / "raw.csv"

        config = {
            **problem.params,
            "context": str(agent_path)
        }

        if args_namespace.deploy:
            run_id = args_namespace.exp.replace("/", "-").removesuffix(".json")
            resume= "allow"
        else:
            run_id = args_namespace.exp.replace("/", "-").removesuffix(".json") + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            resume = "never"
            
        wandb_run = wandb.init(
            entity="plant-rl",
            project="main",
            id = run_id,
            resume=resume,
            notes=str(agent_path),
            config=config,
            settings=wandb.Settings(
                x_stats_disk_paths=("/", "/data"), # So wandb alerts when data dir is near full
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
            is_mock_env = exp.problem.startswith("Mock")
            log(env, glue, wandb_run, s, a, info, is_mock_env=is_mock_env, episode=episode)
            interaction = Interaction(
                o=s,
                a=a,
                t=False,
                r=None,
                extra=info,
            )
            if not exp.problem.startswith("Mock"):
                img_name = save_images(env, dataset_path, images_save_keys)
                await append_csv(chk, env, glue, raw_csv_path, img_name, interaction)

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
            collector.collect('action', env.last_action)
            collector.collect('agent_action', interaction.a)
            collector.collect('reward', interaction.r)
            collector.collect('terminal', interaction.t)
            collector.collect('steps', glue.num_steps)
            for key, value in interaction.extra.items():
                if isinstance(value, np.ndarray):
                    value = value.astype(np.float64)
                elif isinstance(value, pd.DataFrame):
                    for col in value.columns:
                        collector.collect(f"{key}_{col}", value[col].to_numpy().astype(np.float64))
                    continue
                collector.collect(key, value)

            episodic_return = glue.total_reward if interaction.t else None
            episode = chk['episode']
            log(env, glue, wandb_run, interaction.o, interaction.a, interaction.extra, is_mock_env=is_mock_env, r=interaction.r, t=interaction.t, episodic_return=episodic_return, episode=episode)

            if not is_mock_env:
                img_name = save_images(env, dataset_path, images_save_keys)
                await append_csv(chk, env, glue, raw_csv_path, img_name, interaction)

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
                log(env, glue, wandb_run, s, a, info, is_mock_env=is_mock_env)
                interaction = Interaction(
                    o=s,
                    a=a,
                    t=False,
                    r=None,
                    extra=info,
                )
                if not exp.problem.startswith("Mock"):
                    img_name = save_images(env, dataset_path, images_save_keys)
                    await append_csv(chk, env, glue, raw_csv_path, img_name, interaction)

            # Pass exp to backup_and_save if it's not accessible otherwise.
            # Or make backup_and_save a method of a class that holds exp.
            # For now, assuming backup_and_save can access exp if it's defined in main's scope
            # or passed appropriately. If backup_and_save is called from here, it has access to `exp`.
            backup_and_save(exp, collector, idx, args_namespace.save_path)

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
        backup_and_save(exp, collector, idx, args_namespace.save_path)
        wandb_run.finish()

async def append_csv(chk, env, glue, raw_csv_path, img_name, interaction):
    expanded_info = {}
    for key, value in interaction.extra.items():
        if isinstance(value, pd.DataFrame):
            continue
        elif isinstance(value, np.ndarray):
            expanded_info.update(expand(key, value))
        else:
            expanded_info.update(expand(key, value))
    df = pd.DataFrame(
        {
            "time": [env.time],
            "frame": [glue.num_steps],
            **expand("state", interaction.o),
            **expand("action", env.last_action),
            "agent_action": [interaction.a],
            "reward": [interaction.r],
            "terminal": [interaction.t],
            "steps": [glue.num_steps],
            "image_name": [img_name],
            "return": [glue.total_reward if interaction.t else None],
            "episode": [chk['episode']],
            **expanded_info,
        }
    )

    interaction.extra["df"].reset_index(inplace=True)
    interaction.extra["df"]["plant_id"] = interaction.extra["df"].index
    interaction.extra["df"]["frame"] = glue.num_steps
    df = pd.merge(
        df,
        interaction.extra["df"],
        how="left",
        left_on=["frame"],
        right_on=["frame"],
    )
    if await asyncio.to_thread(raw_csv_path.exists):
        # Read only the header of the existing CSV
        async with aiofiles.open(raw_csv_path, mode='r') as f:
            existing_header = (await f.readline()).strip()
        existing_columns = existing_header.split(',')
        new_columns = df.columns.tolist()

        if existing_columns == new_columns:
            # Columns are the same, append without header
            async with aiofiles.open(raw_csv_path, mode='a', newline='') as f:
                await asyncio.to_thread(df.to_csv, f, header=False, index=False)
        else:
            # Columns are different, read old data, backup, concatenate, and write with header
            df_old = await asyncio.to_thread(pd.read_csv, raw_csv_path)
            await asyncio.to_thread(shutil.copy, raw_csv_path, raw_csv_path.with_suffix('.bak'))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
                    category=FutureWarning,
                )
                # pd.concat is generally CPU bound and fast, might not need to_thread
                # but being consistent for now.
                df = await asyncio.to_thread(pd.concat, [df_old, df], ignore_index=True)
            await asyncio.to_thread(df.to_csv, raw_csv_path, index=False)
    else:
        # File does not exist, write with header
        await asyncio.to_thread(df.to_csv, raw_csv_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    device = 'gpu' if args.gpu else 'cpu'
    jax.config.update('jax_platform_name', device)
    
    # Update logging setup based on args if needed, e.g. args.silent
    # This part of logging was already conditional on args.silent (via `prod`)
    # so it might be okay, but good to ensure `prod` is updated if it moves.
    # `prod` calculation:
    prod = 'cdr' in socket.gethostname() or args.silent
    if not prod:
        logger.setLevel(logging.DEBUG)

    asyncio.run(main(args))
