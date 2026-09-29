"""Action-parity and retrain tests for the E23/P1 deployment agents.

E23/P1 re-runs E22/P1's reward comparison on E18+E21+E22 data (plant-data v30,
E22 Z3 cut at day 10 after a lost tray). Like E22/P1 it compares the
**standard** (absolute) reward against the **differential**
reward, which subtracts the same-batch same-day constant-white control:

    R_t = (Δlog A − Δlog A_ctl) − λ·(E_t − E_ctl)/E_ctl

The differential arm's dynamics model regresses the de-confounded growth
*advantage* (``area_reward_diff``, built per row in plant-data with the true
experiment id) rather than the observation delta, so this suite also covers a
model whose output is not ``next_obs - obs``. Zone 5 is no longer a policy arm: it runs a constant
blue + orange-red spectrum at 100 PPFD.

Golden values live in tests/test_data/E23P1/golden_actions.json (dumped by
model-uncertainty-exploration scripts/dump_golden_actions.py --experiment E23).
Checkpoints are staged (gitignored) under checkpoints/E23/P1 — run
experiments/online/E23/P1/ship_checkpoints.sh first.
"""

import datetime as _datetime
import json
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest

from algorithms.ConstantAgent import ConstantAgent
from algorithms.jax.AdaptivePPOPolicy import AdaptivePPOPolicy
from algorithms.jax.PPOPolicy import PPOPolicy
from experiment.ExperimentModel import load as load_experiment

ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = ROOT / "checkpoints" / "E23" / "P1"
CONFIG_DIR = ROOT / "experiments" / "online" / "E23" / "P1"
GOLDEN_PATH = ROOT / "tests" / "test_data" / "E23P1" / "golden_actions.json"

pytestmark = pytest.mark.skipif(
    not CKPT_ROOT.exists(),
    reason="E23/P1 checkpoints not staged; run "
    "experiments/online/E23/P1/ship_checkpoints.sh",
)

# zone -> (checkpoint dir, head)
ZONES = {
    "Z1": ("standard", "ppo_eval"),
    "Z2": ("differential", "ppo_eval"),
    "Z3": ("masked_e21", "ppo_eval"),
    "Z4": ("differential", "ppo_explore"),
}
ADAPTIVE_ZONES = {"differential": "Z4"}
RETRAIN_MODE = {"differential": "analytic_diff"}
CKPT_NAMES = ("differential", "standard", "masked_e21")
ATOL = 1e-4


def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def _ckpt_dir(params: dict) -> str:
    """Which staged checkpoint tree a config points at."""
    for name in CKPT_NAMES:
        if f"/{name}/" in params["checkpoint_path"]:
            return name
    raise AssertionError(f"unrecognised checkpoint_path {params['checkpoint_path']!r}")


def zone_params(zone: str) -> dict:
    params = json.loads((CONFIG_DIR / f"{zone}.json").read_text())["metaParameters"]
    name = _ckpt_dir(params)
    params["checkpoint_path"] = str(CKPT_ROOT / name / "checkpoint")
    if "dataset_npz" in params:
        params["dataset_npz"] = str(CKPT_ROOT / name / "offline_transitions.npz")
    # The real config points at the live /data/plant-rl archive tree. A test
    # retrain must never write there: it would leave test-sized artifacts that
    # collide with (and so suppress) that night's genuine archive.
    params.pop("retrain_archive_dir", None)
    return params


AgentT = TypeVar("AgentT", bound=PPOPolicy)


def make_agent(
    zone: str, cls: type[AgentT] = PPOPolicy, seed: int = 0, **overrides
) -> AgentT:
    return cls((1,), 1, {**zone_params(zone), **overrides}, None, seed)


def plant_extra(areas, ids=None) -> dict:
    ids = list(range(len(areas))) if ids is None else ids
    return {"df": pd.DataFrame({"pot_id": ids, "clean_area": list(areas)})}


def grid_actions(agent, grid) -> list[float]:
    actions = []
    for obs in grid:
        agent._day = int(obs[1]) if len(obs) > 1 else 0
        actions.append(agent._select_action(np.asarray(obs[:1], dtype=np.float32)))
    return actions


class _FrozenDate(_datetime.datetime):
    ordinal = 0

    @classmethod
    def now(cls, tz=None):
        d = _datetime.date.fromordinal(cls.ordinal)
        return _datetime.datetime(d.year, d.month, d.day, 9, 0, tzinfo=tz)


@pytest.fixture
def frozen_date(monkeypatch):
    def freeze(ordinal: int):
        _FrozenDate.ordinal = ordinal
        monkeypatch.setattr("algorithms.jax.PPOPolicy.datetime", _FrozenDate)

    return freeze


class TestConfigs:
    @pytest.mark.parametrize("zone", sorted(ZONES))
    def test_matches_ops_conventions(self, zone):
        p = json.loads((CONFIG_DIR / f"{zone}.json").read_text())["metaParameters"]
        assert (p["action_min"], p["action_max"]) == (0.4, 1.3)
        assert p["action_timestep"] == 720
        assert p["environment"]["observation"] == "log_area"
        assert p["environment"]["action"] == "intensity"
        assert p["hidden_dim"] == 64 and p["activation"] == "tanh"

    @pytest.mark.parametrize("zone", sorted(ZONES))
    def test_head_and_obs_dim_match_the_checkpoint(self, zone):
        p = json.loads((CONFIG_DIR / f"{zone}.json").read_text())["metaParameters"]
        name, head = ZONES[zone]
        assert _ckpt_dir(p) == name
        assert p["policy"] == ("eval" if head == "ppo_eval" else "explore")
        assert p["action_selection"] == ("mean" if head == "ppo_eval" else "sample")
        # All three arms carry the day, so the standard-vs-differential contrast
        # is the reward alone and not a different state space.
        assert p["obs_dim"] == 2

    @pytest.mark.parametrize("name,zone", sorted(ADAPTIVE_ZONES.items()))
    def test_explore_arms_retrain_under_their_own_reward(self, name, zone):
        p = json.loads((CONFIG_DIR / f"{zone}.json").read_text())["metaParameters"]
        assert p["reward_mode"] == RETRAIN_MODE[name]
        assert "dataset_npz" in p and "retrain_archive_dir" in p

    def test_z3_runs_the_e21_pinned_masked_reward(self):
        """Z3's checkpoint is the masked arm gated on 90% of E21 Z11.

        The stock ``masked_log`` gate is pinned to E18 Z11. E21's cohort was ~56%
        larger by day 14, so the E21-pinned gate is much harder — 46.7% of the
        e18e21e22-daily-v30 transitions fall off-target under it (48.1% on
        E22/P1's v29 data) — a deliberate design choice, not a drop-in swap.
        Recomputing the E21 Z11 baseline from the v30 parquet reproduces the
        ``masked_log_e21`` constants bit-for-bit, so the mode is reused as-is.
        """
        p = json.loads((CONFIG_DIR / "Z3.json").read_text())["metaParameters"]
        assert _ckpt_dir(p) == "masked_e21"
        assert p["policy"] == "eval" and p["action_selection"] == "mean"
        assert p["obs_dim"] == 2, "masked modes require the day in the obs"

    def test_z11_is_the_constant_white_control(self):
        """Z11 must reproduce the fixed policy the differential reward references.

        The control tables come from E18 Z11 / E21 Z11 / E22 Z11 running constant
        balanced white at measured intensity ~0.9955, so E23 Z11 has to run the
        same thing (constant_action 1.0) for the reference to mean anything.
        """
        cfg = json.loads((CONFIG_DIR / "Constant11.json").read_text())
        assert cfg["agent"].startswith("Constant")
        m = cfg["metaParameters"]
        assert m["constant_action"] == 1.0
        assert m["environment"]["zone"] == "alliance-zone11"
        assert m["environment"]["action"] == "intensity"
        assert m["action_timestep"] == 720
        # Flash photography must stay on: it is what captures the daily
        # standardized image the offline vision pipeline needs to recover the
        # control's growth. The constant agent ignores online CV, but P1 runs
        # it anyway to log the control's hourly area alongside the agent arms.
        assert m["flash_photography"] is True and m["enforce_night"] is True
        assert m["environment"].get("enable_cv_pipeline") is True
        # No policy keys: it is not a PPO arm.
        assert "checkpoint_path" not in m and "reward_mode" not in m

    def test_z11_continues_its_p0_policy(self):
        """P1's control must be the same zone/policy P0 already ran."""
        p0 = CONFIG_DIR.parent / "P0" / "Constant11.json"
        a = json.loads((CONFIG_DIR / "Constant11.json").read_text())
        b = json.loads(p0.read_text())
        # Only online CV differs: P0 ran it off, P1 turns it on for logging.
        assert a["metaParameters"]["environment"].pop("enable_cv_pipeline") is True
        assert b["metaParameters"]["environment"].pop("enable_cv_pipeline") is False
        assert a == b, "Z11 should carry straight over from E23/P0"

    def test_z5_is_the_constant_orange_red_spectrum(self):
        """Z5 replaces E22's all_data arm with a fixed alternative spectrum.

        Blue + orange-red only, at the same 100 PPFD and the same blue share as
        balanced white (30.9% of photons in 400-500 nm), so a rosette-area
        difference against Z11 is attributable to swapping white for orange-red
        rather than to less blue. Orange-red must stay under its 79 PPFD safe max.

        Goes through the real loader: a bare list in metaParameters is a
        hyperparameter SWEEP, so ``[30.9, 0, 0, 69.1, 0, 0]`` would silently run
        permutation 0 as ``constant_action = 30.9`` (30.9x white, ~3090 PPFD).
        The vector must be wrapped as a one-element sweep, ``[[...]]``.
        """
        exp = load_experiment(str(CONFIG_DIR / "Constant5.json"))
        assert exp.agent.startswith("Constant")
        assert exp.numPermutations() == 1
        m = exp.getPermutation(0)["metaParameters"]
        assert m["environment"]["zone"] == "alliance-zone05"
        assert m["environment"]["action"] == "ppfd6"
        agent = ConstantAgent((1,), 6, m, None, 0)
        spectrum = np.asarray(agent.policy(np.zeros(1)))
        np.testing.assert_allclose(spectrum, [30.90, 0.0, 0.0, 69.10, 0.0, 0.0])
        assert spectrum[:5].sum() == pytest.approx(100.0)
        assert spectrum[3] < 79
        assert m["action_timestep"] == 720
        assert m["flash_photography"] is True and m["enforce_night"] is True
        assert "checkpoint_path" not in m and "reward_mode" not in m

    def test_exploit_arms_declare_no_retrain(self):
        for zone in ("Z1", "Z2", "Z3"):
            p = json.loads((CONFIG_DIR / f"{zone}.json").read_text())["metaParameters"]
            assert "reward_mode" not in p and "dataset_npz" not in p


class TestStagedBuffers:
    @pytest.mark.parametrize("name", sorted(ADAPTIVE_ZONES))
    def test_buffer_matches_the_policy_and_the_data(self, name):
        npz = np.load(CKPT_ROOT / name / "offline_transitions.npz")
        assert npz["obs"].shape[1] == 2, "buffer obs_dim must match the config"
        # Bounds must come from the data, not from the Minari declared space:
        # plant-data derives that from pre-filter stats and its low is 0.00276,
        # which would let the retrain's ClipAction explore far below anything
        # observed. The real operating range is [0.4, 1.3].
        assert 0.38 <= float(npz["act_low"][0]) <= 0.4
        assert 1.29 <= float(npz["act_high"][0]) <= 1.3
        # The differential arm regresses a 1-wide advantage; the standard arm a
        # 1-wide log-area delta. Neither predicts the day.
        assert npz["dyn_target"].shape[1] == 1
        assert npz["ctl_growth"].shape == npz["ctl_energy"].shape == (15,)
        assert np.all(np.isfinite(npz["ctl_growth"]))

    def test_differential_target_subtracts_a_per_experiment_control(self):
        """growth - dyn_target is the control, resolved per (experiment, day).

        Offline rows are de-confounded with each row's OWN experiment's control
        (plant-data has the experiment id), so within a day the subtracted value
        takes at most one distinct value per experiment present — three, for
        E18+E21+E22. It is NOT the pooled ``ctl_growth`` table, which is the
        unweighted mean of the three and is only used for online rows and for
        turning an advantage back into an area.
        """
        npz = np.load(CKPT_ROOT / "differential" / "offline_transitions.npz")
        growth = npz["next_obs"][:, 0] - npz["obs"][:, 0]
        subtracted = growth - npz["dyn_target"][:, 0]
        day = npz["obs"][:, 1].astype(int)
        nd = np.minimum(day + 1, len(npz["ctl_growth"]) - 1)
        for d in np.unique(day):
            vals = subtracted[day == d]
            # At most three experiments contribute a control on any day.
            assert len(np.unique(np.round(vals, 4))) <= 3, f"day {d}"
            # And the pooled table sits between/near them.
            pooled = npz["ctl_growth"][nd[day == d][0]]
            assert abs(vals.mean() - pooled) < 0.08, f"day {d}"

    def test_only_the_adaptive_arm_ships_a_buffer(self):
        """Frozen arms need no replay buffer; shipping one would be dead weight."""
        for name in CKPT_NAMES:
            npz = CKPT_ROOT / name / "offline_transitions.npz"
            assert npz.exists() == (name in ADAPTIVE_ZONES.keys()), name


class TestGoldenParity:
    @pytest.mark.parametrize("zone", sorted(ZONES))
    def test_mean_actions_match_golden(self, zone):
        g = golden()
        name, head = ZONES[zone]
        agent = make_agent(zone, action_selection="mean")
        actions = grid_actions(agent, g[name]["grid"])
        np.testing.assert_allclose(actions, g[name][head]["mean_actions"], atol=ATOL)

    @pytest.mark.parametrize("zone", sorted(ZONES))
    def test_actions_stay_inside_ops_bounds(self, zone):
        g = golden()
        name, _ = ZONES[zone]
        agent = make_agent(zone, action_selection="mean")
        actions = np.asarray(grid_actions(agent, g[name]["grid"]))
        assert np.all(actions >= 0.4) and np.all(actions <= 1.3)


class TestNightlyRetrain:
    @pytest.mark.parametrize("name", sorted(ADAPTIVE_ZONES))
    def test_retrain_completes_and_swaps(self, name):
        """The critical check for the differential arm.

        ``analytic_diff`` is a new reward mode in the vendored mbrl port, paired
        with a dynamics model whose output is an advantage rather than an
        observation delta. ``plan()`` swallows exceptions and still returns to
        phase "idle", so reaching "idle" is NOT evidence of success — only the
        swap counter proves the retrain ran under this reward.
        """
        agent = make_agent(
            ADAPTIVE_ZONES[name],
            cls=AdaptivePPOPolicy,
            retrain_after_hour=0,  # any hour qualifies; deadline window empty
            mbrl={
                "model": {"update_steps": 40},
                "model_steps_per_slice": 20,
                "ppo": {"num_envs": 64, "total_timesteps": 64 * 10 * 4},
                "chunk_updates": 2,
            },
        )
        areas0 = np.full(20, 12.0)
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(areas0))
        agent.step(0.0, np.array([0.55], dtype=np.float32), plant_extra(areas0 * 1.05))
        assert agent._pointer == agent._offline_count + 20

        phases = []
        for _ in range(100):
            agent.plan()
            phases.append(agent._phase)
            if phases[-1] == "idle" and len(phases) > 1:
                break
        assert "model" in phases and "ppo" in phases
        assert agent._phase == "idle" and agent._retrain is None
        assert agent._retrain_count == 1

        action, _ = agent.step(
            0.0, np.array([0.6], dtype=np.float32), plant_extra(areas0 * 1.1)
        )
        assert 0.4 <= action <= 1.3

    def test_online_rows_get_a_dynamics_target(self):
        """Online transitions must carry a target, or the retrain regresses zeros."""
        agent = make_agent(ADAPTIVE_ZONES["differential"], cls=AdaptivePPOPolicy)
        areas0 = np.full(20, 12.0)
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(areas0))
        agent.step(0.0, np.array([0.55], dtype=np.float32), plant_extra(areas0 * 1.05))
        n, p = agent._offline_count, agent._pointer
        rows = agent._buffer_dyn_target[n:p]
        assert rows.shape == (20, 1) and np.all(np.isfinite(rows))
        # Growth minus the control, not the raw growth: the two differ by the
        # day's control growth, which is far from zero.
        growth = agent._buffer_next_obs[n:p, 0] - agent._buffer_obs[n:p, 0]
        assert not np.allclose(rows[:, 0], growth, atol=1e-3)

    def test_checkpoint_roundtrip_restores_targets(self, tmp_path):
        agent = make_agent(ADAPTIVE_ZONES["differential"], cls=AdaptivePPOPolicy)
        areas0 = np.full(20, 12.0)
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(areas0))
        agent.step(0.0, np.array([0.55], dtype=np.float32), plant_extra(areas0 * 1.05))
        n, p = agent._offline_count, agent._pointer
        before = agent._buffer_dyn_target[n:p].copy()

        state = agent.__getstate__()
        restored = make_agent(ADAPTIVE_ZONES["differential"], cls=AdaptivePPOPolicy)
        restored.__setstate__(state)
        after = restored._buffer_dyn_target[n : restored._pointer]
        np.testing.assert_allclose(after, before, atol=1e-6)
