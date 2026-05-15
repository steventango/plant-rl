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
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo paths --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Disable wandb so the loader doesn't try to start a run.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from experiment import ExperimentModel  # noqa: E402  (after sys.path edit)
from problems.registry import getProblem  # noqa: E402
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

N_STEPS = 14   # one entry per mock-env day; covers our 14-day trial window


def _mocktest_config(src_path: Path) -> Path:
    """Produce a temp JSON identical to `src_path` but with problem swapped to
    MockAreaPlantGrowthChamberIntensity and the mock env params injected."""
    cfg = json.loads(src_path.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["problem"] = "MockAreaPlantGrowthChamberIntensity"
    cfg["total_steps"] = N_STEPS
    env = cfg["metaParameters"]["environment"]
    env["dataset_path"] = str(MOCK_DATASET)
    env["experiment"] = MOCK_EXPERIMENT
    env["zone_id"] = MOCK_ZONE_ID
    env["mock_area"] = True
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_mocktest.json", delete=False, dir="/tmp"
    )
    json.dump(cfg, tmp, indent=4)
    tmp.close()
    return Path(tmp.name)


async def _run_one(label: str, src_path: Path):
    print(f"  running {label} via main_real path...")
    cfg_path = _mocktest_config(src_path)
    exp = ExperimentModel.load(str(cfg_path))
    Problem = getProblem(exp.problem)
    problem = Problem(exp, 0, None)
    env = problem.getEnvironment()
    agent = problem.getAgent()

    with tempfile.TemporaryDirectory() as save_dir:
        dataset_path = Path(save_dir)
        env.set_dataset_path(dataset_path)
        glue = AsyncRLGlue(agent, env, dataset_path, images_save_keys=None)

        t_secs, actions = [], []
        interaction = await glue.start()
        t_secs.append(env.time.timestamp())
        actions.append(np.asarray(interaction.a, dtype=float))
        for _ in range(N_STEPS - 1):
            interaction = await glue.step()
            a = np.asarray(interaction.a, dtype=float)
            if not np.all(np.isfinite(a)):
                # Mock dataset has run out of valid samples; stop cleanly.
                break
            t_secs.append(env.time.timestamp())
            actions.append(a)

    return {
        "t_secs": np.array(t_secs),
        "actions": np.stack(actions),
    }


def _to_per_channel(actions: np.ndarray) -> np.ndarray:
    """SequenceAgent/ConstantAgent emit scalar s = PPFD/105; the env multiplies
    by BALANCED_ACTION_105 to get the 6-channel PPFD vector. Reproduce that
    here so the plot shows what physically reaches the lightbar."""
    from utils.constants import BALANCED_ACTION_105
    if actions.ndim == 1 or actions.shape[-1] != 6:
        return np.outer(actions.ravel(), BALANCED_ACTION_105)
    return actions


def _plot(label: str, data: dict):
    actions = _to_per_channel(data["actions"])
    t_hours = (data["t_secs"] - data["t_secs"][0]) / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        if actions[:, i].max() < 1e-6:
            continue
        ax.plot(t_hours, actions[:, i], color=color, marker="o", markersize=4,
                linewidth=1.2, label=name)
    ax.set_xlabel("hours from first step")
    ax.set_ylabel("per-channel PPFD (µmol m⁻² s⁻¹)")
    ax.set_title(f"{label} — actions emitted via main_real path (mock env, {N_STEPS} steps)")
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
        actions = _to_per_channel(data["actions"])
        ppfd = actions[:, :5].sum(axis=1)
        print(f"  {label}: first 5 steps PPFD = {np.round(ppfd[:5], 2).tolist()}; "
              f"min/max over {N_STEPS} steps = {ppfd.min():.1f}/{ppfd.max():.1f}")


if __name__ == "__main__":
    asyncio.run(_main())
