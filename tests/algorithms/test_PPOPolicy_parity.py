"""Action-parity tests for the E20/P1 deployment agents.

Asserts that PPOPolicy / AdaptivePPOPolicy, instantiated from the real
experiments/online/E20/P1 configs, reproduce the golden actions computed in
model-uncertainty-exploration (scripts/dump_golden_actions.py) from the same
checkpoints. Golden values live in tests/test_data/E20P1/golden_actions.json;
the checkpoints themselves are staged (gitignored) under checkpoints/E20/P1 —
run experiments/online/E20/P1/ship_checkpoints.sh first.
"""

import datetime as _datetime
import json
import time
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest

import jax.numpy as jnp

from algorithms.jax.AdaptivePPOPolicy import AdaptivePPOPolicy
from algorithms.jax.PPOPolicy import PPOPolicy

ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = ROOT / "checkpoints" / "E20" / "P1"
CONFIG_DIR = ROOT / "experiments" / "online" / "E20" / "P1"
GOLDEN_PATH = ROOT / "tests" / "test_data" / "E20P1" / "golden_actions.json"

pytestmark = pytest.mark.skipif(
    not CKPT_ROOT.exists(),
    reason="E20/P1 checkpoints not staged; run "
    "experiments/online/E20/P1/ship_checkpoints.sh",
)

# (mode, head) -> zone config exercising that checkpoint/head combination
ZONE_FOR = {
    ("analytic", "ppo_eval"): "Z1",
    ("masked", "ppo_eval"): "Z2",
    ("analytic", "ppo_explore"): "Z3",
    ("masked", "ppo_explore"): "Z4",
}
ADAPTIVE_ZONES = {"analytic": "Z5", "masked": "Z11"}
ATOL = 1e-4


def golden():
    return json.loads(GOLDEN_PATH.read_text())


def zone_params(zone: str) -> dict:
    config = json.loads((CONFIG_DIR / f"{zone}.json").read_text())
    params = config["metaParameters"]
    mode = "masked" if "masked" in params["checkpoint_path"] else "analytic"
    params["checkpoint_path"] = str(CKPT_ROOT / mode / "checkpoint")
    if "dataset_npz" in params:
        params["dataset_npz"] = str(CKPT_ROOT / mode / "offline_transitions.npz")
    return params


AgentT = TypeVar("AgentT", bound=PPOPolicy)


def make_agent(
    zone: str, cls: type[AgentT] = PPOPolicy, seed: int = 0, **overrides
) -> AgentT:
    params = {**zone_params(zone), **overrides}
    return cls((1,), 1, params, None, seed)


def plant_extra(areas, ids=None) -> dict:
    """A per-poll ``extra`` with a per-plant df, like the real env supplies."""
    ids = list(range(len(areas))) if ids is None else ids
    return {"df": pd.DataFrame({"pot_id": ids, "clean_area": list(areas)})}


def grid_actions(agent, grid) -> list[float]:
    """Feed each golden grid observation through the public action path."""
    actions = []
    for obs in grid:
        agent._day = int(obs[1]) if len(obs) > 1 else 0
        action = agent._select_action(np.asarray(obs[:1], dtype=np.float32))
        actions.append(action)
    return actions


class _FrozenDate(_datetime.datetime):
    """datetime stub whose now() lands on a fixed ordinal date."""

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


class TestMeanActionParity:
    @pytest.mark.parametrize(("mode", "head"), sorted(ZONE_FOR))
    def test_mean_actions_match_golden(self, mode, head):
        g = golden()
        agent = make_agent(ZONE_FOR[(mode, head)], action_selection="mean")
        actions = grid_actions(agent, g[mode]["grid"])
        np.testing.assert_allclose(actions, g[mode][head]["mean_actions"], atol=ATOL)

    @pytest.mark.parametrize("mode", ["analytic", "masked"])
    def test_log_std_matches_golden(self, mode):
        g = golden()
        for head in ("ppo_explore", "ppo_eval"):
            agent = make_agent(ZONE_FOR[(mode, head)])
            std = float(np.exp(np.asarray(agent._network.actor.log_std.get_value()))[0])
            assert std == pytest.approx(g[mode][head]["std"], abs=ATOL)


class TestSampledActionParity:
    @pytest.mark.parametrize("mode", ["analytic", "masked"])
    def test_sampled_actions_match_golden(self, mode, frozen_date):
        g = golden()
        zone = ZONE_FOR[(mode, "ppo_explore")]
        agent = make_agent(zone, seed=g["seed"])
        assert agent._action_selection == "sample"
        expected = g[mode]["ppo_explore"]["sampled_actions"]
        for j, ordinal in enumerate(g["day_ordinals"]):
            frozen_date(ordinal)
            actions = grid_actions(agent, g[mode]["grid"])
            np.testing.assert_allclose(actions, [row[j] for row in expected], atol=ATOL)

    def test_same_day_sampling_is_reproducible(self, frozen_date):
        frozen_date(739100)
        agent = make_agent("Z3")
        first, _ = agent.start(np.array([0.5], dtype=np.float32), {})
        second, _ = agent.start(np.array([0.5], dtype=np.float32), {})
        assert first == second


class TestDayIndexSynthesis:
    def test_day_appended_and_clamped(self):
        agent = make_agent("Z4")
        agent._day = 7
        obs = np.asarray(agent._assemble_obs(np.array([0.3], dtype=np.float32)))
        assert obs.tolist() == pytest.approx([0.3, 7.0])
        agent._day = 20  # deployment runs past the 14-day training episode
        obs = np.asarray(agent._assemble_obs(np.array([0.3], dtype=np.float32)))
        assert obs[1] == 14.0

    def test_day_counter_survives_pickle(self, setup_checkpoint_test, tmpdir):
        params = zone_params("Z4")

        def collect(agent):
            agent.start(np.array([0.4], dtype=np.float32), {})
            agent.step(0.0, np.array([0.5], dtype=np.float32), {})

        original, loaded = setup_checkpoint_test(
            tmpdir,
            params,
            PPOPolicy,
            observations=(1,),
            actions=1,
            init_func=collect,
        )
        assert loaded._day == original._day == 1


class TestAdaptivePPOPolicy:
    @pytest.mark.parametrize("mode", sorted(ADAPTIVE_ZONES))
    def test_initial_action_matches_explore_golden(self, mode):
        g = golden()
        agent = make_agent(
            ADAPTIVE_ZONES[mode], cls=AdaptivePPOPolicy, action_selection="mean"
        )
        actions = grid_actions(agent, g[mode]["grid"])
        np.testing.assert_allclose(
            actions, g[mode]["ppo_explore"]["mean_actions"], atol=ATOL
        )
        # The restored ENN world model predicts finite deltas.
        x = agent._model.normalize_input(jnp.zeros(agent._obs_dim + 1))
        assert bool(jnp.all(jnp.isfinite(agent._model.predict_mean(x))))

    @pytest.mark.parametrize("mode", sorted(ADAPTIVE_ZONES))
    def test_nightly_retrain_completes(self, mode):
        if mode == "masked":
            pytest.skip(
                "E20/P1 is retired: its Z11 config uses reward_mode 'masked', which "
                "was renamed/removed when the reward moved to next-area gating "
                "(now masked_log/masked_growth). Its checkpoint was trained on the "
                "removed area-surplus reward, so this retrain is unreproducible. "
                "E21/P1 covers the current masked_log path."
            )
        agent = make_agent(
            ADAPTIVE_ZONES[mode],
            cls=AdaptivePPOPolicy,
            retrain_after_hour=0,  # any hour qualifies; deadline window empty
            mbrl={
                "model": {"update_steps": 40},
                "model_steps_per_slice": 20,
                "ppo": {"num_envs": 64, "total_timesteps": 64 * 10 * 4},
                "chunk_updates": 2,
            },
        )
        # One daily poll of 20 plants (equal Δ, so the IQR trim keeps all)
        # yields 20 per-plant transitions.
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
        assert agent._last_retrain_date is not None
        # plan() swallows exceptions and still returns to "idle", so returning to
        # idle is NOT evidence of success — assert the swap actually happened.
        assert agent._retrain_count == 1

        # Same night: no second cycle.
        agent.plan()
        assert agent._phase == "idle"

        action, _ = agent.step(
            0.0, np.array([0.6], dtype=np.float32), plant_extra(areas0 * 1.1)
        )
        assert 0.4 <= action <= 1.3

    def test_per_plant_collection_one_transition_per_pot(self):
        agent = make_agent(ADAPTIVE_ZONES["analytic"], cls=AdaptivePPOPolicy)
        n0 = agent._offline_count
        # Below the IQR-trim threshold, so every matched pot is kept and the
        # per-pot day-to-day mapping is exact.
        ids = list(range(7))
        a0 = np.exp(np.linspace(-1.0, 0.5, 7))
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(a0, ids))
        assert agent._pointer == n0  # start collects nothing (no previous poll)

        agent.step(0.0, np.array([0.5], dtype=np.float32), plant_extra(a0 * 1.04, ids))
        assert agent._pointer - n0 == 7  # one transition per pot, matched by id

        obs = agent._buffer_obs[n0 : agent._pointer, 0]
        nxt = agent._buffer_next_obs[n0 : agent._pointer, 0]
        # Rows are per-plant log-areas, chained per pot: next = obs + log(1.04).
        np.testing.assert_allclose(np.sort(obs), np.sort(np.log(a0)), atol=1e-4)
        np.testing.assert_allclose(nxt - obs, np.log(1.04), atol=1e-4)

    def test_reward_iqr_drops_extreme_growth(self):
        agent = make_agent(ADAPTIVE_ZONES["analytic"], cls=AdaptivePPOPolicy)
        n0 = agent._offline_count
        ids = list(range(20))
        a0 = np.full(20, 10.0)
        a1 = a0 * 1.02  # 18 plants grow normally...
        a1[0] = a0[0] * 5.0  # ...one wild CV over-detection
        a1[1] = a0[1] * 0.1  # ...one collapse
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(a0, ids))
        agent.step(0.0, np.array([0.5], dtype=np.float32), plant_extra(a1, ids))
        assert agent._pointer - n0 == 18  # the two Δlog outliers are trimmed

    def test_dead_plants_and_nan_are_dropped(self):
        agent = make_agent(ADAPTIVE_ZONES["analytic"], cls=AdaptivePPOPolicy)
        n0 = agent._offline_count
        ids = [0, 1, 2, 3, 4]
        a0 = np.array([10.0, 20.0, 0.0, np.nan, 15.0])  # pot2 dead, pot3 CV failure
        a1 = np.array([11.0, 22.0, 0.0, np.nan, 16.0])
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(a0, ids))
        agent.step(0.0, np.array([0.5], dtype=np.float32), plant_extra(a1, ids))
        assert agent._pointer - n0 == 3  # only pots 0, 1, 4 valid on both polls

    def test_checkpoint_roundtrip(self, setup_checkpoint_test, tmpdir):
        params = zone_params("Z5")

        def collect(agent):
            a0 = np.exp(np.linspace(-0.5, 0.5, 16))
            agent.start(np.array([0.4], dtype=np.float32), plant_extra(a0))
            agent.step(0.0, np.array([0.5], dtype=np.float32), plant_extra(a0 * 1.03))

        original, loaded = setup_checkpoint_test(
            tmpdir,
            params,
            AdaptivePPOPolicy,
            observations=(1,),
            actions=1,
            init_func=collect,
        )
        assert loaded._pointer == original._pointer == original._offline_count + 16
        assert loaded._day == original._day
        probe = jnp.asarray(np.array([0.5], dtype=np.float32))
        assert loaded._policy_action(probe) == pytest.approx(
            original._policy_action(probe), abs=1e-6
        )

    @pytest.mark.slow
    def test_plan_slice_bounded_at_production_hparams(self):
        """One plan() slice must stay small so it can never delay a poll."""
        agent = make_agent(
            "Z5",
            cls=AdaptivePPOPolicy,
            retrain_after_hour=0,
            mbrl={"chunk_updates": 2, "model_steps_per_slice": 500},
        )
        agent.plan()  # idle -> setup
        assert agent._phase == "model"
        agent.plan()  # compile + first model slice
        t0 = time.time()
        agent.plan()  # steady-state model slice
        model_slice_s = time.time() - t0
        assert model_slice_s < 10, f"model slice took {model_slice_s:.1f}s"

        while agent._phase == "model":
            agent.plan()
        assert agent._phase == "ppo"
        agent.plan()  # compile + first PPO chunk
        t0 = time.time()
        agent.plan()  # steady-state PPO chunk
        ppo_slice_s = time.time() - t0
        assert ppo_slice_s < 10, f"ppo chunk took {ppo_slice_s:.1f}s"
        agent._reset_retrain()


class TestRegistry:
    def test_agent_names_resolve(self):
        from algorithms.registry import getAgent

        assert getAgent("PPOPolicy1") is PPOPolicy
        assert getAgent("AdaptivePPOPolicy5") is AdaptivePPOPolicy
        assert getAgent("AdaptivePPOPolicy11") is AdaptivePPOPolicy


class TestConfigsConsistent:
    def test_configs_match_golden_bounds(self):
        g = golden()
        for zone in ["Z1", "Z2", "Z3", "Z4", "Z5", "Z11", "Z12"]:
            params = zone_params(zone)
            assert params["action_min"] == g["action_min"]
            assert params["action_max"] == g["action_max"]
            expected_obs_dim = 2 if "masked" in params["checkpoint_path"] else 1
            assert params["obs_dim"] == expected_obs_dim


class TestZ12Derisk:
    """Z12 is the hardware rehearsal for the masked adaptive arm Z11 — it must
    not drift from it (mirrors Z11 to derisk the masked path on hardware)."""

    def test_z12_mirrors_z11_except_zone_and_timezone(self):
        z11 = json.loads((CONFIG_DIR / "Z11.json").read_text())
        z12 = json.loads((CONFIG_DIR / "Z12.json").read_text())
        assert z12["agent"] == "AdaptivePPOPolicy12"

        m11, m12 = z11["metaParameters"], z12["metaParameters"]
        assert m12["environment"]["zone"] == "alliance-zone12"
        assert m12["environment"]["timezone"] == "America/Edmonton"
        assert m12["timezone"] == "America/Edmonton"
        assert m12["reward_mode"] == "masked" and m12["obs_dim"] == 2
        # Z12 runs the full observation pipeline: CV must not be disabled.
        assert m12["environment"].get("enable_cv_pipeline", True)

        excluded = {"zone", "timezone"}
        env11 = {k: v for k, v in m11["environment"].items() if k not in excluded}
        env12 = {k: v for k, v in m12["environment"].items() if k not in excluded}
        assert env11 == env12
        rest11 = {k: v for k, v in m11.items() if k not in ("environment", "timezone")}
        rest12 = {k: v for k, v in m12.items() if k not in ("environment", "timezone")}
        assert rest11 == rest12

    def test_z12_agent_constructs_with_edmonton_tz(self, frozen_date):
        agent = make_agent("Z12", cls=AdaptivePPOPolicy)
        assert str(agent._tz) == "America/Edmonton"
        assert agent._action_selection == "sample"
        frozen_date(739100)
        action, _ = agent.start(np.array([0.5], dtype=np.float32), {})
        assert 0.4 <= action <= 1.3
