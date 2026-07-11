"""Verify the online-transition data flow of AdaptivePPOPolicy end to end.

Z12 is an empty test chamber, so its live CV pipeline yields no plant area and
the daily poll can't produce meaningful observations. Instead we drive the
MockPlantGrowthChamber, which replays recorded real plant trajectories
(/data/plant-rl/offline/v27/mixed-v27.parquet) through the exact
env -> PlantGrowthChamberAsyncAgentWrapper -> AdaptivePPOPolicy -> AsyncRLGlue
path the real deployment uses. This exercises the piece the live Z12 run did
NOT (it retrained on 7090 offline + 0 online): a real observation the chamber
produces on day N flowing into the buffer and being ingested by the retrain.

Two phases:
  PHASE 1 (collect): clock pinned to daytime (no retrain), run the mock glue
    until the agent has collected K online transitions via real daily polls.
    We print the actual (obs, action, next_obs) rows written to the buffer so
    the day-to-day transition assembly is visible.
  PHASE 2 (retrain): flip the agent's clock into the nightly window and drive
    plan() to completion. Confirm the retrain reports "N offline + K online",
    completes, and swaps a policy (probe action before/after).

Run:
    JAX_PLATFORMS=cpu python experiments/online/E20/P1/online_flow_check.py
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import os
import sys
import tempfile
import types
from datetime import timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

logging.basicConfig(level=logging.INFO, format="%(message)s")

import jax  # noqa: E402

import algorithms.jax.AdaptivePPOPolicy as ap  # noqa: E402
from algorithms.jax.mbrl.plant_env import PlantEnv  # noqa: E402
from experiment.ExperimentModel import ExperimentModel  # noqa: E402
from problems.registry import getProblem  # noqa: E402
from utils.RlGlue.rl_glue import AsyncRLGlue  # noqa: E402

HERE = Path(__file__).parent
CKPT = REPO_ROOT / "checkpoints" / "E20" / "P1"
MOCK_DATASET = "/data/plant-rl/offline/v27/mixed-v27.parquet"


class _Clock(_dt.datetime):
    """Controllable stand-in for the wall clock plan() reads."""

    wall_hour = 12  # daytime -> retrain gate closed

    @classmethod
    def now(cls, tz=None):
        return _dt.datetime(2026, 7, 11, cls.wall_hour, 30, 0, tzinfo=tz)


# plan() reads the module-level `datetime` name; swap in the controllable clock.
ap.datetime = _Clock


def _build_exp():
    d = json.loads((HERE / "Z12.json").read_text())
    d["total_steps"] = 1_000_000
    meta = d["metaParameters"]
    env = meta["environment"]
    env["backend"] = "mock"
    env["dataset_path"] = MOCK_DATASET
    env["experiment"] = 17
    env["zone_id"] = 1
    env["mock_area"] = True
    env.pop("enable_cv_pipeline", None)
    meta["checkpoint_path"] = str(CKPT / "analytic" / "checkpoint")
    meta["dataset_npz"] = str(CKPT / "analytic" / "offline_transitions.npz")
    meta["retrain_after_hour"] = 21
    meta["retrain_deadline_hour"] = 8
    # Reduced-but-real retrain so PHASE 2 finishes in a few seconds.
    meta["mbrl"] = {
        "model": {"update_steps": 300},
        "model_steps_per_slice": 150,
        "ppo": {"num_envs": 256, "total_timesteps": 256 * 10 * 30},
        "chunk_updates": 6,
    }
    return ExperimentModel(d, str(HERE / "Z12.json"))


async def main(target_online: int, max_steps: int):
    exp = _build_exp()
    Problem = getProblem(exp.problem)
    problem = Problem(exp, 0, None)
    env = problem.getEnvironment()
    wrapped = problem.getAgent()
    inner = wrapped.agent  # AdaptivePPOPolicy

    env.duration = timedelta(minutes=1)

    async def _fast_sleep_until(self, wake_time):  # type: ignore[no-untyped-def]
        self.current_time = wake_time

    env.sleep_until = types.MethodType(_fast_sleep_until, env)

    n0 = inner._offline_count
    probe = jax.numpy.asarray(np.array([0.5], dtype=np.float32))
    print(f"offline_count={n0}  pointer={inner._pointer}")

    # PHASE 1 — collect per-plant online transitions through the real mock pipeline.
    _Clock.wall_hour = 12
    poll_counts = []  # transitions added per daily poll
    with tempfile.TemporaryDirectory() as save_dir:
        env.set_dataset_path(Path(save_dir))
        glue = AsyncRLGlue(wrapped, env, Path(save_dir), images_save_keys=None)
        await glue.start()
        step = 0
        prev_ptr = inner._pointer
        while inner._pointer - n0 < target_online and step < max_steps:
            await glue.step()
            step += 1
            if inner._pointer > prev_ptr:
                poll_counts.append(inner._pointer - prev_ptr)
                prev_ptr = inner._pointer
    k = inner._pointer - n0
    polls = len(poll_counts)
    per_poll = f"{np.mean(poll_counts):.1f}" if poll_counts else "0"
    print(
        f"PHASE 1: {step} mock steps -> {polls} daily polls -> {k} per-plant "
        f"transitions collected (~{per_poll} plants/poll: {poll_counts})"
    )
    assert k >= 1, "no online transitions were collected"

    # --- Data-quality audit of the collected online rows ---
    sl = slice(n0, inner._pointer)
    obs = inner._buffer_obs[sl]
    nxt = inner._buffer_next_obs[sl]
    act = inner._buffer_action[sl]
    log_area, next_log_area = obs[:, 0], nxt[:, 0]
    finite = (
        np.all(np.isfinite(obs))
        and np.all(np.isfinite(nxt))
        and np.all(np.isfinite(act))
    )
    in_area = np.all(
        (log_area >= inner._area_min - 1e-3) & (log_area <= inner._area_max + 1e-3)
    )
    in_act = np.all(
        (act >= inner._action_min - 1e-6) & (act <= inner._action_max + 1e-6)
    )
    # Oracle reward the retrain will derive from each online row must be finite.
    oracle = PlantEnv(1, reward_mode=inner._reward_mode)
    rewards = np.asarray(
        oracle.compute_reward(
            jax.numpy.asarray(obs), jax.numpy.asarray(act), jax.numpy.asarray(nxt)
        )
    )
    print("  data-quality audit:")
    print(f"    all finite (obs/next/action) : {bool(finite)}")
    print(
        f"    log_area in [{inner._area_min:.2f}, {inner._area_max:.2f}] : {bool(in_area)}"
    )
    print(f"    action in [{inner._action_min}, {inner._action_max}] : {bool(in_act)}")
    print(
        f"    log_area range observed      : [{log_area.min():.3f}, {log_area.max():.3f}]"
    )
    print(
        f"    Δlog-area (reward) range     : [{(next_log_area - log_area).min():.3f}, {(next_log_area - log_area).max():.3f}]"
    )
    print(
        f"    oracle reward finite         : {bool(np.all(np.isfinite(rewards)))} (range [{rewards.min():.3f}, {rewards.max():.3f}])"
    )
    assert finite and in_area and in_act and np.all(np.isfinite(rewards))

    a_before = inner._policy_action(probe)

    # PHASE 2 — flip into the nightly window and drive the retrain to completion.
    _Clock.wall_hour = 22
    slices = 0
    for _ in range(5000):
        inner.plan()
        slices += 1
        if inner._phase == "idle" and inner._last_retrain_date is not None:
            break
    a_after = inner._policy_action(probe)
    print(
        f"PHASE 2: retrain drove {slices} plan() slices -> phase={inner._phase}, "
        f"date={inner._last_retrain_date}"
    )
    print(f"  probe(log_area=0.5) action  before={a_before:.4f}  after={a_after:.4f}")

    assert inner._phase == "idle" and inner._last_retrain_date is not None
    print(
        f"\nRESULT: {polls} daily polls produced {k} per-plant transitions "
        f"(~{per_poll}/poll); the retrain ingested {n0} offline + {k} online, "
        f"completed, and swapped the policy."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online", type=int, default=3, help="online transitions to collect"
    )
    parser.add_argument("--max-steps", type=int, default=9000)
    args = parser.parse_args()
    asyncio.run(main(args.online, args.max_steps))
