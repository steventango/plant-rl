import asyncio
import os
import shutil
import sys
import warnings

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
from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Identity, Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict
from ml_instrumentation.metadata import attach_metadata

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import expand, log
from utils.preempt import TimeoutHandler
from rlglue import AsyncRlGlue # Assuming AsyncRlGlue is now directly in rlglue
# Interaction class might be obsolete if glue.step() returns a simple tuple/dict

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
    # This function will be replaced by direct logic
    pass

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
        collector.set_experiment_id(idx)
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
        dataset_path = Path('/data') / agent_path  / f"z{env.zone.identifier}"
        images_save_keys = problem.exp_params.get("image_save_keys", default_save_keys)
        dataset_path.mkdir(parents=True, exist_ok=True)
        raw_csv_path = dataset_path / "raw.csv"

        config = {
            **problem.params,
            "context": str(agent_path)
        }

        if args.deploy:
            run_id = args.exp.replace("/", "-").removesuffix(".json")
            resume= "allow"
        else:
            run_id = args.exp.replace("/", "-").removesuffix(".json") + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
            # Assuming glue.start() returns a 3-tuple (obs, action, info_dict)
            # and Interaction class is no longer needed or adapted.
            # This example focuses on glue.step() return value changes.
            obs_start, action_start, info_start = await glue.start()
            episode = chk['episode']
            log(env, glue, wandb_run, obs_start, action_start, info_start, episode=episode)
            # For append_csv, we need to ensure it can handle the new tuple structure or an adapted Interaction-like object
            # For now, assuming append_csv is adapted or its direct call here is simplified/modified.
            # Let's construct a temporary dict mimicking old Interaction for append_csv if strictly needed.
            interaction_for_csv_start = {'o': obs_start, 'a': action_start, 't': False, 'r': None, 'extra': info_start}
            if not exp.problem.startswith("Mock"):
                img_name = save_images(env, dataset_path, images_save_keys)
                append_csv(chk, env, glue, raw_csv_path, img_name, interaction_for_csv_start)

        for step in range(glue.total_steps, exp.total_steps):
            collector.next_frame()
            if problem.exp_params.get("checkpoint", True):
                if current_save_task and not current_save_task.done():
                    current_save_task.cancel()
                    logger.debug(f"Cancelled previous checkpoint save at step {step}")
                current_save_task = asyncio.create_task(asyncio.to_thread(chk.save))

            observation, reward, terminated, truncated, info = await glue.step() # RLGlue returns 5-tuple
            collector.collect('time', env.time.timestamp())
            collector.collect('state', observation)
            collector.collect('action', env.last_action) # env.last_action seems to be the intended "environment action"
            collector.collect('agent_action', info.get('action', -1)) # Agent's chosen action from info
            collector.collect('reward', reward)
            collector.collect('terminal', terminated)
            collector.collect('steps', glue.num_steps)
            for key, value in info.items(): # info is the info_dict from the 5-tuple
                if isinstance(value, np.ndarray):
                    value = value.astype(np.float64)
                elif isinstance(value, pd.DataFrame):
                    for col in value.columns:
                        collector.collect(f"{key}_{col}", value[col].to_numpy().astype(np.float64))
                    continue
                collector.collect(key, value)

            episodic_return = glue.total_reward if terminated else None
            episode = chk['episode']
            # log function needs to be adapted for the new 5-tuple structure
            # log(env, glue, wandb_run, observation, info.get('action', -1), info, reward, terminated, episodic_return, episode)
            # For now, using a simplified call or assuming log is adapted.
            log(env, glue, wandb_run, observation, info.get('action', -1), info, reward, terminated, episodic_return, episode)


            if not exp.problem.startswith("Mock"):
                # Construct a temporary dict for append_csv if it relies on old Interaction structure
                interaction_for_csv_step = {'o': observation, 'a': info.get('action', -1), 't': terminated, 'r': reward, 'extra': info}
                img_name = save_images(env, dataset_path, images_save_keys)
                append_csv(chk, env, glue, raw_csv_path, img_name, interaction_for_csv_step)

            if terminated or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
                # collect some data
                collector.collect('return', glue.total_reward)
                collector.collect('episode', chk['episode'])
                collector.collect('steps', glue.num_steps)

                # track how many episodes are completed (cutoff is counted as termination for this count)
                chk['episode'] += 1

                # compute the average time-per-step in ms
                avg_time = 1000 * (time.time() - start_time) / (step + 1)
                fps = step / (time.time() - start_time)

                episode = chk['episode'] # This is chk['episode'] before increment for this log
                logger.debug(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')

                # Assuming glue.start() returns a 3-tuple (obs, action, info_dict)
                obs_start_new_ep, action_start_new_ep, info_start_new_ep = await glue.start()
                # log function needs to be adapted for the new 3-tuple structure from start()
                log(env, glue, wandb_run, obs_start_new_ep, action_start_new_ep, info_start_new_ep)
                # For append_csv, construct temporary dict if needed
                interaction_for_csv_new_ep_start = {'o': obs_start_new_ep, 'a': action_start_new_ep, 't': False, 'r': None, 'extra': info_start_new_ep}
                if not exp.problem.startswith("Mock"):
                    img_name = save_images(env, dataset_path, images_save_keys)
                    append_csv(chk, env, glue, raw_csv_path, img_name, interaction_for_csv_new_ep_start)
            
            # Save logic replacement
            context = exp.buildSaveContext(idx, base=args.save_path)
            save_db_path = context.resolve('results.db')
            if os.path.exists(save_db_path):
                shutil.copy(save_db_path, str(save_db_path) + '.bak')
            meta = getParamsAsDict(exp, idx)
            meta |= {'seed': run}
            attach_metadata(save_db_path, idx, meta)
            collector.merge(save_db_path)
            # collector.close() will be called outside the loop

        collector.reset() # This might be redundant if collector.close() is called, depending on Collector's behavior

        env.close()

        # Wait for all checkpoint save tasks to complete before exiting
        if current_save_task and not current_save_task.done():
            logger.debug("Waiting for the last checkpoint save task to complete...")
            await current_save_task
            logger.debug("Last checkpoint save task completed.")

        # ------------
        # -- Saving (final, outside loop) --
        # ------------
        # The save logic is now per step inside the loop.
        # A final save might still be desired, or collector.close() handles finalization.
        # Based on main.py, collector.close() handles the final merge.
        collector.close()
        wandb_run.finish()

def append_csv(chk, env, glue, raw_csv_path, img_name, interaction_dict): # interaction_dict is the temporary dict
    expanded_info = {}
    # interaction_dict['extra'] is the info_dict from glue.step()
    for key, value in interaction_dict['extra'].items():
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
            **expand("state", interaction_dict['o']), # Use 'o' from dict
            **expand("action", env.last_action),
            "agent_action": [interaction_dict['a']], # Use 'a' from dict
            "reward": [interaction_dict['r']], # Use 'r' from dict
            "terminal": [interaction_dict['t']], # Use 't' from dict
            "steps": [glue.num_steps],
            "image_name": [img_name],
            "return": [glue.total_reward if interaction_dict['t'] else None],
            "episode": [chk['episode']],
            **expanded_info,
        }
    )

    # Ensure interaction_dict['extra'] (info_dict) has 'df' if this logic is to be kept
    if "df" in interaction_dict['extra']:
        interaction_dict['extra']["df"].reset_index(inplace=True)
        interaction_dict['extra']["df"]["plant_id"] = interaction_dict['extra']["df"].index
        interaction_dict['extra']["df"]["frame"] = glue.num_steps
        df = pd.merge(
            df,
            interaction_dict['extra']["df"],
        how="left",
        left_on=["frame"],
        right_on=["frame"],
    )
    if raw_csv_path.exists():
        df_old = pd.read_csv(raw_csv_path)
        shutil.copy(raw_csv_path, raw_csv_path.with_suffix('.bak'))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
                category=FutureWarning
            )
            df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(raw_csv_path, index=False)


if __name__ == "__main__":
    asyncio.run(main())
