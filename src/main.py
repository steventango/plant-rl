import asyncio
import json
import os
import sys

from environments.PlantGrowthChamber.zones import serialize_zone

sys.path.append(os.getcwd())
import argparse
import logging
import socket
from pathlib import Path

import jax
import numpy as np

import wandb
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.logger import WandbAlertHandler, log

from utils.preempt import TimeoutHandler
from utils.RlGlue.rl_glue import AsyncRLGlue

logger = logging.getLogger("plant-data")


async def main():
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

    logging.basicConfig(
        level=logging.ERROR,
        format="[%(asctime)s] %(levelname)s:%(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    prod = "cdr" in socket.gethostname() or args.silent
    if not prod:
        logger.setLevel(logging.DEBUG)

    for idx in args.idxs:
        run_id = args.exp.replace("/", "-").removesuffix(".json")

        wandb_run = wandb.init(
            entity="anffanychen-university-of-alberta",
            project="plant-data-collection",
            id=run_id,
            resume="allow",
            config={},
            settings=wandb.Settings(
                x_stats_disk_paths=(
                    "/",
                    "/data",
                ),  # So wandb alerts when data dir is near full
                init_timeout=180,
            ),
        )

        # Set up wandb alert handler
        handler = WandbAlertHandler(wandb_run)
        logger.addHandler(handler)

        env = None
        chk = None

        try:
            # ----------------------
            # -- Experiment Def'n --
            # ----------------------
            timeout_handler = TimeoutHandler()
            exp = ExperimentModel.load(args.exp)

            Problem = getProblem(exp.problem)

            chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
            loaded = chk.load_if_exists()
            if loaded:
                logger.info("Loaded checkpoint")
            timeout_handler.before_cancel(chk.save)

            run = exp.getRun(idx)

            # set random seeds accordingly
            np.random.seed(run)

            # build stateful things and attach to checkpoint
            problem = chk.build("p", lambda: Problem(exp, idx, None))
            agent = chk.build("a", problem.getAgent)
            env = chk.build("e", problem.getEnvironment)

            context = exp.buildSaveContext(idx, base=args.save_path)
            agent_path = Path(context.resolve()).relative_to("results")
            dataset_path = Path("/data/plant-data-collection/") / agent_path / env.zone.identifier
            env.set_dataset_path(dataset_path)
            images_save_keys = problem.exp_params.get("image_save_keys")

            config = {
                **problem.params,
                "context": str(agent_path),
                "zone": serialize_zone(env.zone),
            }
            wandb_run.config.update(config, allow_val_change=True)

            def glue_builder():
                assert env is not None, (
                    "Environment must be initialized before creating glue."
                )
                return AsyncRLGlue(
                    agent,
                    env,
                    dataset_path,
                    images_save_keys=images_save_keys,
                )

            glue = chk.build("glue", glue_builder)
            chk.initial_value("episode", 0)

            # save config to dataset
            if not dataset_path.exists():
                dataset_path.mkdir(parents=True, exist_ok=True)
            config_path = dataset_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            # if we haven't started yet, then make the first interaction
            if glue.total_steps == 0:
                # Load partial checkpoint if specified (fine-tuning)
                load_params = problem.exp_params.get("load", None)
                if isinstance(load_params, dict):
                    loaded_chk = Checkpoint(
                        exp,
                        0,
                        base_path=args.checkpoint_path,
                        load_path=load_params["path"],
                    )
                    loaded_chk.load()
                    chk.load_from_checkpoint(loaded_chk, load_params.get("config"))

                interaction = await glue.start()
                episode = chk["episode"]
                log(
                    env,
                    glue,
                    wandb_run,
                    interaction.o,
                    interaction.a,
                    interaction.extra,
                    episode=episode,
                )

            for step in range(glue.total_steps, exp.total_steps):
                interaction = await glue.step()

                episodic_return = glue.total_reward if interaction.t else None
                episode = chk["episode"]
                log(
                    env,
                    glue,
                    wandb_run,
                    interaction.o,
                    interaction.a,
                    interaction.extra,
                    r=interaction.r,
                    t=interaction.t,
                    episodic_return=episodic_return,
                    episode=episode,
                )


                if interaction.t or (
                    exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff
                ):
                    chk["episode"] += 1

                    interaction = await glue.start()
                    log(
                        env,
                        glue,
                        wandb_run,
                        interaction.o,
                        interaction.a,
                        interaction.extra,
                    )
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            # ------------
            # -- Saving --
            # ------------
            if env is not None and hasattr(env, "close"):
                await env.close()
            if chk is not None:
                try:
                    chk.save()
                except Exception:
                    logger.exception("Failed to save checkpoint")
                    raise
        wandb_run.finish()

if __name__ == "__main__":
    asyncio.run(main())
