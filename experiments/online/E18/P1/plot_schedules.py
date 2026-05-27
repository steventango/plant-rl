"""Plot and self-test the six E18/P1 deploy configs (Z1 / Z2 / Z3 / Z4 / Z5 / Z11).

Reads the six JSON configs, walks the wrapper's polling logic across a
simulated 14-day timeline at 1-min resolution, and renders:
  * PPFD vs wrapper-local time (1 panel per zone, 14 days stacked)
  * Mean daily PPFD profile (1 panel; collapses 14 days onto a 24 h axis)
  * Power vs time, cumulative energy curves
  * Channel activations per zone (which LEDs are on, per slot)

The simulation directly drives `PlantGrowthChamberAsyncAgentWrapper` with a
real `SequenceAgent` / `ConstantAgent` instance, so any drift between the
plan and the deploy JSONs will surface here.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algorithms.PlantGrowthChamberAsyncAgentWrapper import (  # noqa: E402
    PlantGrowthChamberAsyncAgentWrapper,
)
from algorithms.registry import getAgent  # noqa: E402
from utils.constants import BALANCED_ACTION_100  # noqa: E402

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

CONFIGS = {
    "Z1 power-law ramp": HERE / "PowerLawRamp1.json",
    "Z2 within-day parabola": HERE / "Parabolic2.json",
    "Z3 flat low-DLI": HERE / "ConstantLow3.json",
    "Z4 70% ramp": HERE / "SeventyPercentRamp4.json",
    "Z5 late-biased ramp": HERE / "LateRamp5.json",
    "Z11 constant 100": HERE / "Constant11.json",
}

ZONE_COLOR = {
    "Z1 power-law ramp": "tab:blue",
    "Z2 within-day parabola": "tab:orange",
    "Z3 flat low-DLI": "tab:purple",
    "Z4 70% ramp": "tab:red",
    "Z5 late-biased ramp": "tab:brown",
    "Z11 constant 100": "tab:green",
}

ENV_STEP_MIN = 1
TOTAL_MIN = 14 * 24 * 60  # 14-day simulation
START = datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc)


def lights_on_power(ppfd_total: np.ndarray) -> np.ndarray:
    """E18/P0.1 pooled fit, lights-on; returns 7.21 W baseline when zero."""
    out = np.where(
        ppfd_total > 0,
        9.71 + 0.164 * np.power(np.maximum(ppfd_total, 1e-9), 1.19),
        7.21,
    )
    return out


def lights_on_only_power(ppfd_total: np.ndarray) -> np.ndarray:
    """Lights-on plug power, zero when off. Used for the lights-on-only Wh sum
    that matches the README's 6623 / 6573 / 8241 Wh figures."""
    return np.where(
        ppfd_total > 0, 9.71 + 0.164 * np.power(np.maximum(ppfd_total, 1e-9), 1.19), 0.0
    )


SAFE_MIN = np.array([5.0, 5.0, 5.0, 4.0, 5.0, 0.6679889999999996])


def _promote_to_balanced(action) -> np.ndarray:
    """Mirror PlantGrowthChamberIntensity.step's scalar -> 6-channel scaling.

    SequenceAgent / ConstantAgent return scalar `s`; the env multiplies by
    BALANCED_ACTION_100. Wrapper-enforced (night/dawn/flash) actions are
    already 6-vectors and pass through.
    """
    arr = np.asarray(action, dtype=float)
    if arr.ndim == 0:
        return float(arr) * BALANCED_ACTION_100
    if arr.shape == (6,):
        return arr
    return float(arr.ravel()[0]) * BALANCED_ACTION_100


def _build_agent(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())
    meta = cfg["metaParameters"]
    AgentCls = getAgent(cfg["agent"])
    inner = AgentCls(
        observations=(1,),
        actions=1,
        params=meta,
        collector=None,  # type: ignore[arg-type]
        seed=0,
    )
    return PlantGrowthChamberAsyncAgentWrapper(inner), meta


async def _simulate_zone(cfg_path: Path) -> dict[str, np.ndarray]:
    wrapper, meta = _build_agent(cfg_path)

    actions = np.zeros((TOTAL_MIN, 6))
    obs = np.zeros((1,))
    extra: dict[str, Any] = {"env_time": START.timestamp()}

    action, _ = await wrapper.start(obs, extra)
    actions[0] = _promote_to_balanced(action)

    for t in range(1, TOTAL_MIN):
        extra["env_time"] = (START + timedelta(minutes=t * ENV_STEP_MIN)).timestamp()
        action, _ = await wrapper.step(0.0, obs, extra)
        actions[t] = _promote_to_balanced(action)

    ppfd = actions[:, :5].sum(axis=1)
    # n_chan = number of channels whose desired PPFD clears its safe_min
    # (matches Calibration.get_calibrated_action's `active` mask without
    # calling it 20k+ times).
    n_chan = (actions >= SAFE_MIN).sum(axis=1).astype(int)
    n_chan[ppfd <= 0] = 0

    return {"ppfd": ppfd, "n_chan": n_chan, "meta": meta}


def simulate_all() -> dict[str, dict[str, np.ndarray]]:
    results = {}
    for label, path in CONFIGS.items():
        print(f"  simulating {label}...")
        results[label] = asyncio.run(_simulate_zone(path))
    return results


def plot_ppfd_timeline(results) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), sharex=True)
    axes = np.atleast_1d(axes)
    t_hours = np.arange(TOTAL_MIN) / 60.0
    for ax, (label, data) in zip(axes, results.items(), strict=False):
        ax.plot(t_hours, data["ppfd"], color=ZONE_COLOR[label], linewidth=0.7)
        ax.set_ylabel("PPFD\nµmol m⁻² s⁻¹")
        ax.set_title(f"{label}  (peak {data['ppfd'].max():.0f} PPFD)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(data["ppfd"].max() * 1.1, 140))
        # daily grid markers
        for d in range(15):
            ax.axvline(d * 24, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
    axes[-1].set_xlabel("hours from simulation start")
    fig.suptitle("E18/P1 — 14-day PPFD timeline (simulated wrapper output)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "schedule_ppfd_timeline.pdf")
    plt.close(fig)


def plot_daily_profile(results) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data in results.items():
        # mean PPFD per minute-of-day across 14 days
        per_min = data["ppfd"].reshape(14, 24 * 60).mean(axis=0)
        ax.plot(
            np.arange(24 * 60) / 60.0,
            per_min,
            color=ZONE_COLOR[label],
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("wrapper-local hour of day")
    ax.set_ylabel("PPFD (mean across 14 days)")
    ax.set_title("E18/P1 — mean daily PPFD profile per zone")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(np.arange(0, 25, 3))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "schedule_daily_profile.pdf")
    plt.close(fig)


def plot_power_and_cumulative(results) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    t_hours = np.arange(TOTAL_MIN) / 60.0
    for label, data in results.items():
        p = lights_on_power(data["ppfd"])
        cum = np.cumsum(p) / 60.0  # W·min → Wh (each step is 1 min)
        axes[0].plot(
            t_hours, p, color=ZONE_COLOR[label], linewidth=0.5, label=label, alpha=0.85
        )
        axes[1].plot(
            t_hours,
            cum,
            color=ZONE_COLOR[label],
            linewidth=1.5,
            label=f"{label} (final {cum[-1]:.0f} Wh)",
        )
    axes[0].set_ylabel("plug power (W)")
    axes[0].set_title("E18/P1 — instantaneous plug power vs time")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("hours from simulation start")
    axes[1].set_ylabel("cumulative energy (Wh)")
    axes[1].set_title("Cumulative lights-on energy over the 14-day window")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "schedule_power_cumulative.pdf")
    plt.close(fig)


def self_test(results) -> bool:
    """Hard assertions: each zone must match its planned daily energy and PPFD range.

    `cum_Wh` here is *lights-on only* (excludes the 7.21 W idle baseline that
    runs continuously) and includes the fixed 1-min daily flash. It should land
    about 5.4 Wh above the README's schedule-only 14-day energy table.
    """
    expected = {
        "Z1 power-law ramp": dict(
            min_ppfd=38, max_ppfd=124, cum_min=6600, cum_max=6650
        ),
        "Z2 within-day parabola": dict(
            min_ppfd=40, max_ppfd=130, cum_min=6550, cum_max=6600
        ),
        "Z3 flat low-DLI": dict(min_ppfd=40, max_ppfd=78, cum_min=6540, cum_max=6570),
        "Z4 70% ramp": dict(
            min_ppfd=40, max_ppfd=95, cum_min=5790, cum_max=5825
        ),
        "Z5 late-biased ramp": dict(
            min_ppfd=40, max_ppfd=130, cum_min=6570, cum_max=6600
        ),
        "Z11 constant 100": dict(min_ppfd=40, max_ppfd=100, cum_min=8230, cum_max=8260),
    }
    ok = True
    print()
    print(f"{'zone':<28s} min_PPFD  max_PPFD  cum_Wh    n_channels_unique  pass")
    for label, data in results.items():
        lights_on = data["ppfd"] > 0.5
        ppfd_on = data["ppfd"][lights_on]
        if not lights_on.any():
            print(f"  {label}: NO LIGHTS-ON SAMPLES (failure)")
            ok = False
            continue
        cum_wh = float(np.cumsum(lights_on_only_power(data["ppfd"]))[-1] / 60.0)
        exp = expected[label]
        # Round PPFD to nearest 0.5 to avoid floating-point noise on min/max
        min_p = ppfd_on.min()
        max_p = ppfd_on.max()
        passes = (
            abs(min_p - exp["min_ppfd"]) <= 1.5
            and abs(max_p - exp["max_ppfd"]) <= 1.5
            and exp["cum_min"] <= cum_wh <= exp["cum_max"]
        )
        n_chan_unique = sorted(set(int(x) for x in data["n_chan"][lights_on]))
        print(
            f"  {label:<28s} {min_p:6.2f}    {max_p:6.2f}    {cum_wh:7.1f}   {n_chan_unique}     {'PASS' if passes else 'FAIL'}"
        )
        if not passes:
            print(f"      expected {exp}")
            ok = False
    return ok


def main() -> int:
    print("Simulating 14-day timeline for all six configs...")
    results = simulate_all()
    print("Rendering PDF plots...")
    plot_ppfd_timeline(results)
    plot_daily_profile(results)
    plot_power_and_cumulative(results)
    print(f"  -> {FIG_DIR}/schedule_ppfd_timeline.pdf")
    print(f"  -> {FIG_DIR}/schedule_daily_profile.pdf")
    print(f"  -> {FIG_DIR}/schedule_power_cumulative.pdf")
    ok = self_test(results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
