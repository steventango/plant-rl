import asyncio
import os
import shutil
import sys

sys.path.append(os.getcwd())
import argparse
import logging
import socket
import time
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity
from PyExpUtils.results.sqlite import saveCollector

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import log
from utils.preempt import TimeoutHandler
from utils.RlGlue.rl_glue import AsyncRLGlue

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
parser.add_argument(
    "-d",
    "--deploy",
    action="store_true",
    default=False,
    help="Allows for easily restarting logging for crashed runs in deployment. If the run already exists, then wandb will resume logging to the same run.",
)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------


device = "gpu" if args.gpu else "cpu"
jax.config.update("jax_platform_name", device)

logging.basicConfig(
    level=logging.ERROR,
    format="[%(asctime)s] %(levelname)s:%(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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


def backup_and_save(exp, collector, idx, base):
    context = exp.buildSaveContext(idx, base=base)
    db_file = context.resolve("results.db")
    db_file_bak = context.resolve("results.db.bak")
    if os.path.exists(db_file):
        shutil.copy(db_file, db_file_bak)
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
        problem = chk.build("p", lambda: Problem(exp, idx, collector))
        agent = chk.build("a", problem.getAgent)
        env = chk.build("e", problem.getEnvironment)

        context = exp.buildSaveContext(idx, base=args.save_path)
        agent_path = Path(context.resolve()).relative_to("results")
        dataset_path = Path("/data") / agent_path / env.zone.identifier
        images_save_keys = problem.exp_params.get("image_save_keys")

        glue = chk.build(
            "glue", lambda: AsyncRLGlue(agent, env, dataset_path, images_save_keys)
        )
        env.glue = glue
        chk.initial_value("episode", 0)

        config = {**problem.params, "context": str(agent_path)}

        if args.deploy:
            run_id = args.exp.replace("/", "-").removesuffix(".json")
            resume = "allow"
        else:
            run_id = (
                args.exp.replace("/", "-").removesuffix(".json")
                + "-"
                + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            )
            resume = "never"

        wandb_run = wandb.init(
            entity="plant-rl",
            project="main",
            id=run_id,
            resume=resume,
            notes=str(agent_path),
            config=config,
            settings=wandb.Settings(
                x_stats_disk_paths=(
                    "/",
                    "/data",
                ),  # So wandb alerts when data dir is near full
            ),
        )

        try:
            # Run the experiment
            start_time = time.time()
            is_mock_env = exp.problem.startswith("Mock")

            # if we haven't started yet, then make the first interaction
            if glue.total_steps == 0:
                interaction = await glue.start()
                episode = chk["episode"]
                log(
                    env,
                    glue,
                    wandb_run,
                    interaction.o,
                    interaction.a,
                    interaction.extra,
                    is_mock_env=is_mock_env,
                    episode=episode,
                )

            for step in range(glue.total_steps, exp.total_steps):
                collector.next_frame()

                interaction = await glue.step()
                collector.collect("time", env.time.timestamp())
                collector.collect("state", interaction.o)
                collector.collect("action", env.last_action)
                collector.collect("agent_action", interaction.a)
                collector.collect("reward", interaction.r)
                collector.collect("terminal", interaction.t)
                collector.collect("steps", glue.num_steps)
                for key, value in interaction.extra.items():
                    if isinstance(value, np.ndarray):
                        value = value.astype(np.float64)
                    elif isinstance(value, pd.DataFrame):
                        for col in value.columns:
                            collector.collect(
                                f"{key}_{col}", value[col].to_numpy().astype(np.float64)
                            )
                        continue
                    collector.collect(key, value)

                episodic_return = glue.total_reward if interaction.t else None
                episode = chk["episode"]
                log(
                    env,
                    glue,
                    wandb_run,
                    interaction.o,
                    interaction.a,
                    interaction.extra,
                    is_mock_env=is_mock_env,
                    r=interaction.r,
                    t=interaction.t,
                    episodic_return=episodic_return,
                    episode=episode,
                )

                if interaction.t or (
                    exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff
                ):
                    # collect some data
                    collector.collect("return", glue.total_reward)
                    collector.collect("episode", chk["episode"])
                    collector.collect("steps", glue.num_steps)

                    # track how many episodes are completed (cutoff is counted as termination for this count)
                    chk["episode"] += 1

                    # compute the average time-per-step in ms
                    avg_time = 1000 * (time.time() - start_time) / (step + 1)
                    fps = step / (time.time() - start_time)

                    episode = chk["episode"]
                    logger.debug(
                        f"{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}"
                    )

                    interaction = await glue.start()
                    log(
                        env,
                        glue,
                        wandb_run,
                        interaction.o,
                        interaction.a,
                        interaction.extra,
                        is_mock_env=is_mock_env,
                    )

                backup_and_save(exp, collector, idx, args.save_path)

            collector.reset()

        finally:
            # ------------
            # -- Saving --
            # ------------
            backup_and_save(exp, collector, idx, args.save_path)
            await env.close()
            chk.save()
            wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main())
