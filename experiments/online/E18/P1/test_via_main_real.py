"""Run the Z1/Z2/Z3 deploy configs through the same Problem + Env + RLGlue
path that main_real uses, but against the mock chamber (no wandb, no
hardware, no checkpointing) — purely as a pre-deploy smoke test.

We:
  1. Load the deploy JSON via `ExperimentModel` (same loader main_real uses).
  2. Override `problem` to `MockAreaPlantGrowthChamberIntensity` and inject
     `dataset_path` / `experiment` / `zone_id` so the Mock env replays a real
     14-day chunk of E17 data from `/data/plant-rl/offline/v27/mixed-v27.parquet`.
  3. Construct the Problem, env, and wrapped agent exactly as
     `main_real.py:121-137` does.
  4. Drive `AsyncRLGlue` for `min(total_steps, len(unique_wall_times))` steps,
     recording the action vector emitted at each step.
  5. Render one PDF per zone showing the action.* channels over wall-clock
     time — the actual sequence that would have been sent to the lightbar.

Run:
    python experiments/online/E18/P1/test_via_main_real.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import types
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo paths --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Disable wandb so the loader doesn't try to start a run.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from experiment.ExperimentModel import ExperimentModel  # noqa: E402  (after sys.path edit)
from problems.registry import getProblem  # noqa: E402
from utils.constants import BALANCED_ACTION_105  # noqa: E402
from utils.RlGlue.rl_glue import AsyncRLGlue  # noqa: E402

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

MOCK_DATASET = Path("/data/plant-rl/offline/v27/mixed-v27.parquet")
MOCK_EXPERIMENT = 17
MOCK_ZONE_ID = 1   # only used for filtering the mock dataset

CONFIGS = [
    ("Z1 power-law ramp", HERE / "PowerLawRamp1.json"),
    ("Z2 within-day parabola", HERE / "Parabolic2.json"),
    ("Z3 constant 105", HERE / "Constant3.json"),
]

CHANNEL_NAMES = ["blue", "cool_white", "warm_white", "orange_red", "red", "far_red"]
CHANNEL_COLORS = ["tab:blue", "tab:cyan", "goldenrod", "tab:orange", "tab:red", "tab:brown"]

# Simulate the full 14-day deploy window at 1-min env-step granularity. The
# flash is a 1-min event (wrapper checks `minute == 59`), so coarser sim
# steps would *miss* the flash entirely - 1-min ticks match the real
# chamber. 14 days = 20160 env steps per zone.
SIM_STEP_MIN = 1
SIM_DAYS = 14
N_STEPS = SIM_DAYS * 24 * 60 // SIM_STEP_MIN


def _mocktest_exp(src_path: Path) -> ExperimentModel:
    """Load the deploy JSON, swap to the Mock problem and inject the mock env
    params, return an in-memory ExperimentModel (no tempfile required)."""
    d = json.loads(src_path.read_text())
    d["problem"] = "MockAreaPlantGrowthChamberIntensity"
    d["total_steps"] = N_STEPS
    env = d["metaParameters"]["environment"]
    env["dataset_path"] = str(MOCK_DATASET)
    env["experiment"] = MOCK_EXPERIMENT
    env["zone_id"] = MOCK_ZONE_ID
    env["mock_area"] = True
    return ExperimentModel(d, str(src_path))


def _promote_to_six_channel(a_raw) -> np.ndarray:
    """Mirror PlantGrowthChamberIntensity.step's scalar -> 6-channel scaling.

    SequenceAgent/ConstantAgent emit scalar s; the env multiplies by
    BALANCED_ACTION_105. Wrapper-enforced (night/flash) actions are already
    6-vectors and pass through unchanged.
    """
    arr = np.asarray(a_raw, dtype=float)
    if arr.shape == (6,):
        return arr
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.ravel()[0] if arr.size == 1 else arr) * BALANCED_ACTION_105
    raise ValueError(f"unexpected action shape {arr.shape!r}")


async def _run_one(label: str, src_path: Path):
    print(f"  running {label} via main_real path...")
    exp = _mocktest_exp(src_path)
    Problem = getProblem(exp.problem)
    problem = Problem(exp, 0, None)
    env = problem.getEnvironment()
    agent = problem.getAgent()

    # Make the mock env tick at SIM_STEP_MIN granularity instead of jumping
    # to the next dataset record (calendar days apart). Plant data goes stale
    # across the tick - we're testing action emission, not plant dynamics.
    env.duration = timedelta(minutes=SIM_STEP_MIN)

    async def _fast_sleep_until(self, wake_time):  # type: ignore[no-untyped-def]
        self.current_time = wake_time

    env.sleep_until = types.MethodType(_fast_sleep_until, env)

    with tempfile.TemporaryDirectory() as save_dir:
        dataset_path = Path(save_dir)
        env.set_dataset_path(dataset_path)
        glue = AsyncRLGlue(agent, env, dataset_path, images_save_keys=None)

        t_secs, actions = [], []
        interaction = await glue.start()
        t_secs.append(env.time.timestamp())
        actions.append(_promote_to_six_channel(interaction.a))
        for _ in range(N_STEPS - 1):
            interaction = await glue.step()
            a = _promote_to_six_channel(interaction.a)
            if not np.all(np.isfinite(a)):
                # Mock dataset has run out of valid samples; stop cleanly.
                break
            t_secs.append(env.time.timestamp())
            actions.append(a)

    return {
        "t_secs": np.array(t_secs),
        "actions": np.stack(actions),
    }


def _plot(label: str, data: dict):
    actions = data["actions"]  # already normalised to 6-channel by _run_one
    t_hours = (data["t_secs"] - data["t_secs"][0]) / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS, strict=False)):
        if actions[:, i].max() < 1e-6:
            continue
        ax.plot(t_hours, actions[:, i], color=color, marker="o", markersize=4,
                linewidth=1.2, label=name)
    ax.set_xlabel("hours from first step  (env tick = " + str(SIM_STEP_MIN) + " min)")
    ax.set_ylabel("per-channel PPFD (µmol m⁻² s⁻¹)")
    ax.set_title(
        f"{label} — actions emitted via main_real path "
        f"({SIM_DAYS} sim-days at {SIM_STEP_MIN}-min env steps, {N_STEPS} total)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / f"actions_via_main_real_{label.split()[0].lower()}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"    -> {out.name}")


async def _main():
    results = {}
    for label, src in CONFIGS:
        results[label] = await _run_one(label, src)
    print("\nRendering action-over-time PDFs...")
    for label, data in results.items():
        _plot(label, data)

    print("\nPer-step action summary (PPFD totals over first 5 channels):")
    for label, data in results.items():
        actions = data["actions"]
        ppfd = actions[:, :5].sum(axis=1)
        lights_on = ppfd > 0.5
        print(f"  {label}: lights_on fraction = {lights_on.mean():.2%}; "
              f"min/max lights-on PPFD = {ppfd[lights_on].min():.1f}/{ppfd[lights_on].max():.1f}; "
              f"unique daytime PPFD levels ≈ {sorted({round(float(x)) for x in ppfd[lights_on]})}")


if __name__ == "__main__":
    asyncio.run(_main())
