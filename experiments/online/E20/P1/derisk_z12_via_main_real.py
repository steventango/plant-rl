"""Derisk the Z12 hardware test: prove the nightly retrain cannot break acting.

Runs the real Z12 deploy config through the same Problem + Env + AsyncRLGlue
path as main_real, against the mock chamber, with the AdaptivePPOPolicy
nightly retrain FORCED ON from the first plan() call (retrain_after_hour=0)
so the whole cycle interleaves with live env stepping — exactly the hazard the
real Z12 run needs derisked. Asserts:

  1. The retrain state machine completes (model -> ppo -> swap -> idle) while
     the glue keeps stepping.
  2. No glue.step() is stalled longer than MAX_STEP_STALL_S — a plan() slice
     holds the RlGlue lock, so this bounds the worst-case delay a 09:00 poll
     could see.
  3. Every daytime action emitted before, during, and after the retrain stays
     within [action_min, action_max] * BALANCED_ACTION_100.
  4. Pickling the agent mid-retrain (what the :33 checkpoint thread does)
     succeeds and drops the in-flight cycle cleanly.

By default the retrain uses reduced-but-realistic hyperparameters so the
script finishes in a few minutes; pass --full to run the production 10k-step
model update + 10M-timestep PPO retrain (~15-20 min on CPU, the most faithful
pre-deploy rehearsal).

Run:
    python experiments/online/E20/P1/derisk_z12_via_main_real.py [--full]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
import tempfile
import time
import types
from datetime import timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from experiment.ExperimentModel import ExperimentModel  # noqa: E402
from problems.registry import getProblem  # noqa: E402
from utils.constants import BALANCED_ACTION_100  # noqa: E402
from utils.RlGlue.rl_glue import AsyncRLGlue  # noqa: E402

HERE = Path(__file__).parent
CKPT_ROOT = REPO_ROOT / "checkpoints" / "E20" / "P1"

MOCK_DATASET = Path("/data/plant-rl/offline/v27/mixed-v27.parquet")
MOCK_EXPERIMENT = 17
MOCK_ZONE_ID = 1

SIM_STEP_MIN = 1
SIM_DAYS = 3
N_STEPS = SIM_DAYS * 24 * 60 // SIM_STEP_MIN

# A plan() slice holds the RlGlue lock, so the worst-case step stall is one
# slice — including the one-off JIT compile inside the first PPO chunk. The
# chamber steps once a minute, so anything comfortably below that is harmless.
MAX_STEP_STALL_S = 30.0

REDUCED_MBRL = {
    "model": {"update_steps": 2000},
    "model_steps_per_slice": 500,
    "ppo": {"num_envs": 512, "total_timesteps": 512 * 10 * 100},
    "chunk_updates": 8,
}


def _derisk_exp(full: bool) -> ExperimentModel:
    d = json.loads((HERE / "Z12.json").read_text())
    d["problem"] = "PlantGrowthChamber"
    d["total_steps"] = N_STEPS
    meta = d["metaParameters"]
    env = meta["environment"]
    env["backend"] = "mock"
    env["dataset_path"] = str(MOCK_DATASET)
    env["experiment"] = MOCK_EXPERIMENT
    env["zone_id"] = MOCK_ZONE_ID
    env["mock_area"] = True
    env.pop("enable_cv_pipeline", None)

    meta["checkpoint_path"] = str(CKPT_ROOT / "analytic" / "checkpoint")
    meta["dataset_npz"] = str(CKPT_ROOT / "analytic" / "offline_transitions.npz")
    # Fire the nightly cycle immediately (real wall clock) so it overlaps the
    # mock run; the empty [deadline, after) window disables the morning abort.
    meta["retrain_after_hour"] = 0
    if not full:
        meta["mbrl"] = REDUCED_MBRL
    return ExperimentModel(d, str(HERE / "Z12.json"))


def _scalar_ppfd(a_raw) -> float:
    arr = np.asarray(a_raw, dtype=float)
    if arr.shape == (6,):
        return float(arr[:5].sum())
    return float(arr.ravel()[0]) * float(BALANCED_ACTION_100[:5].sum())


async def _main(full: bool):
    exp = _derisk_exp(full)
    Problem = getProblem(exp.problem)
    problem = Problem(exp, 0, None)
    env = problem.getEnvironment()
    wrapped_agent = problem.getAgent()
    inner = wrapped_agent.agent  # AdaptivePPOPolicy

    env.duration = timedelta(minutes=SIM_STEP_MIN)

    async def _fast_sleep_until(self, wake_time):  # type: ignore[no-untyped-def]
        self.current_time = wake_time

    env.sleep_until = types.MethodType(_fast_sleep_until, env)

    daytime_ppfd: list[float] = []
    step_stalls: list[float] = []
    phases_seen: set[str] = set()
    retrain_done_at_step: int | None = None
    mid_retrain_pickle_ok = False

    with tempfile.TemporaryDirectory() as save_dir:
        dataset_path = Path(save_dir)
        env.set_dataset_path(dataset_path)
        glue = AsyncRLGlue(wrapped_agent, env, dataset_path, images_save_keys=None)

        interaction = await glue.start()
        for step in range(N_STEPS - 1):
            t0 = time.monotonic()
            interaction = await glue.step()
            step_stalls.append(time.monotonic() - t0)

            phases_seen.add(inner._phase)
            ppfd = _scalar_ppfd(interaction.a)
            if ppfd > 0.5:
                daytime_ppfd.append(ppfd)

            if inner._phase != "idle" and not mid_retrain_pickle_ok:
                # What the :33 checkpoint thread does, mid-cycle.
                restored = pickle.loads(pickle.dumps(inner))
                assert restored._phase == "idle" and restored._retrain is None
                assert restored._pointer == inner._pointer
                mid_retrain_pickle_ok = True
                print(f"  [step {step}] mid-retrain pickle OK (phase={inner._phase})")

            if (
                retrain_done_at_step is None
                and inner._last_retrain_date is not None
                and inner._phase == "idle"
                and "ppo" in phases_seen
            ):
                retrain_done_at_step = step
                print(f"  [step {step}] nightly retrain completed and swapped")

            if not np.all(np.isfinite(np.asarray(interaction.a, dtype=float))):
                print(f"  [step {step}] mock dataset exhausted; stopping")
                break

    stalls = np.asarray(step_stalls)
    print("\n=== Derisk summary ===")
    print(f"steps driven:            {len(stalls)}")
    print(f"phases seen:             {sorted(phases_seen)}")
    print(f"retrain completed:       step {retrain_done_at_step}")
    print(
        f"mid-retrain pickle:      {'OK' if mid_retrain_pickle_ok else 'NOT EXERCISED'}"
    )
    print(
        f"step stalls (s):         mean {stalls.mean():.3f} | p99 "
        f"{np.percentile(stalls, 99):.3f} | max {stalls.max():.3f}"
    )
    if daytime_ppfd:
        lo, hi = min(daytime_ppfd), max(daytime_ppfd)
        print(f"daytime PPFD range:      [{lo:.1f}, {hi:.1f}]")

    assert retrain_done_at_step is not None, "retrain never completed"
    assert {"model", "ppo"} <= phases_seen, f"phases missing: {phases_seen}"
    assert mid_retrain_pickle_ok, "mid-retrain pickle path not exercised"
    assert stalls.max() < MAX_STEP_STALL_S, (
        f"a glue.step stalled {stalls.max():.1f}s (> {MAX_STEP_STALL_S}s) — a plan() "
        f"slice is holding the lock too long; lower chunk_updates/model_steps_per_slice"
    )
    lights_on_bounds = (
        40.0 * 0.99 <= min(daytime_ppfd) and max(daytime_ppfd) <= 130.0 * 1.01
    )
    assert lights_on_bounds, (
        f"daytime PPFD out of [40, 130]: {min(daytime_ppfd)}..{max(daytime_ppfd)}"
    )
    print("\nDERISK OK — plan() cannot break acting at these settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="use production retrain hyperparameters (~15-20 min)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.full))
