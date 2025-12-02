import os  # type: ignore
import sys

from tqdm import tqdm

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
from utils.logger import log
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
    seed = run + params.get("experiment", {}).get("seed_offset", 0)

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
    exp_path = Path(context.resolve())

    config = {**problem.params, "context": str(agent_path)}

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
        s, a, info = glue.start()
        log(env, glue, wandb_run, s, a, info)

    for step in (pbar := tqdm(range(glue.total_steps, exp.total_steps))):
        collector.next_frame()
        chk.maybe_save()
        interaction = glue.step()
        log(
            env,
            glue,
            wandb_run,
            interaction.o,
            interaction.a,
            interaction.extra,
            r=interaction.r,
            t=interaction.t,
            episodic_return=glue.total_reward if interaction.t else None,
            episode=chk["episode"] if interaction.t else None,
        )

        collector.collect("reward", interaction.r)
        collector.collect("episode", chk["episode"])
        collector.collect("steps", glue.num_steps)
        collector.collect("action", interaction.a)

        if interaction.t or (
            exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff
        ):
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # Collect total reward
            collector.collect("return", glue.total_reward)

            # track how many episodes are completed (cutoff is counted as termination for this count)
            chk["episode"] += 1

            # compute the average time-per-step in ms
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            episode = chk["episode"]
            logger.debug(
                f"{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}"
            )
            pbar.set_description(
                f"Episodes: {episode}, Return: {glue.total_reward:.3f}"
            )

            s, a, info = glue.start()
            log(env, glue, wandb_run, s, a, info)

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)

    # Save final model (InAC-specific)
    if hasattr(agent, "actor_critic") and hasattr(agent, "optimizers"):
        from algorithms.nn.inac.agent.base import save

        save_path = exp_path / str(idx) / "parameters"
        save_path.mkdir(parents=True, exist_ok=True)
        save(agent.actor_critic, agent.optimizers, save_path)  # type: ignore
        logger.info(f"Saved final model to {save_path}")

    # Save checkpoint
    chk.save()
    logger.info("Checkpoint saved")
