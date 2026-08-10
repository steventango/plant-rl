"""Action-parity and retrain tests for the E21/P1 deployment agents.

Asserts that the four E21/P1 zone agents, instantiated from their real configs,
reproduce the golden actions computed in model-uncertainty-exploration
(scripts/dump_golden_actions.py --experiment E21) from the same visu-v28
checkpoints, and that the adaptive explore arms actually complete a nightly
retrain under the current (next-area gated, v28-constant) masked_log reward.

Golden values live in tests/test_data/E21P1/golden_actions.json; the
checkpoints are staged (gitignored) under checkpoints/E21/P1 — run
experiments/online/E21/P1/ship_checkpoints.sh first.
"""

import datetime as _datetime
import json
from pathlib import Path
from typing import TypeVar

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from algorithms.jax.AdaptivePPOPolicy import AdaptivePPOPolicy
from algorithms.jax.PPOPolicy import PPOPolicy

ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = ROOT / "checkpoints" / "E21" / "P1"
CONFIG_DIR = ROOT / "experiments" / "online" / "E21" / "P1"
GOLDEN_PATH = ROOT / "tests" / "test_data" / "E21P1" / "golden_actions.json"

pytestmark = pytest.mark.skipif(
    not CKPT_ROOT.exists(),
    reason="E21/P1 checkpoints not staged; run "
    "experiments/online/E21/P1/ship_checkpoints.sh",
)

ZONES = {
    "Z1": ("analytic", "ppo_eval"),
    "Z2": ("masked_log", "ppo_eval"),
    "Z3": ("analytic", "ppo_explore"),
    "Z4": ("masked_log", "ppo_explore"),
}
ADAPTIVE_ZONES = {"analytic": "Z3", "masked_log": "Z4"}
ATOL = 1e-4


def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def zone_params(zone: str) -> dict:
    params = json.loads((CONFIG_DIR / f"{zone}.json").read_text())["metaParameters"]
    mode = "masked_log" if "masked_log" in params["checkpoint_path"] else "analytic"
    params["checkpoint_path"] = str(CKPT_ROOT / mode / "checkpoint")
    if "dataset_npz" in params:
        params["dataset_npz"] = str(CKPT_ROOT / mode / "offline_transitions.npz")
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


class TestGoldenParity:
    @pytest.mark.parametrize("zone", sorted(ZONES))
    def test_mean_actions_match_golden(self, zone):
        g = golden()
        mode, head = ZONES[zone]
        agent = make_agent(zone, action_selection="mean")
        actions = grid_actions(agent, g[mode]["grid"])
        np.testing.assert_allclose(actions, g[mode][head]["mean_actions"], atol=ATOL)

    @pytest.mark.parametrize("zone", sorted(ZONES))
    def test_log_std_matches_golden(self, zone):
        g = golden()
        mode, head = ZONES[zone]
        agent = make_agent(zone)
        std = float(np.exp(np.asarray(agent._network.actor.log_std.get_value()))[0])
        assert std == pytest.approx(g[mode][head]["std"], abs=ATOL)

    @pytest.mark.parametrize("zone", ["Z3", "Z4"])
    def test_sampled_actions_match_golden(self, zone, frozen_date):
        g = golden()
        mode, head = ZONES[zone]
        agent = make_agent(zone, seed=g["seed"])
        assert agent._action_selection == "sample"
        expected = g[mode][head]["sampled_actions"]
        for j, ordinal in enumerate(g["day_ordinals"]):
            frozen_date(ordinal)
            actions = grid_actions(agent, g[mode]["grid"])
            np.testing.assert_allclose(actions, [r[j] for r in expected], atol=ATOL)


class TestObservationHandling:
    def test_masked_log_zones_use_day_index(self):
        for zone in ("Z2", "Z4"):
            agent = make_agent(zone)
            assert agent._obs_dim == 2
            agent._day = 7
            obs = np.asarray(agent._assemble_obs(np.array([0.3], dtype=np.float32)))
            assert obs.tolist() == pytest.approx([0.3, 7.0])
            agent._day = 25  # deployment outruns the 14-day training episode
            assert (
                np.asarray(agent._assemble_obs(np.array([0.3], np.float32)))[1] == 14.0
            )

    def test_analytic_zones_are_one_dimensional(self):
        for zone in ("Z1", "Z3"):
            agent = make_agent(zone)
            assert agent._obs_dim == 1
            obs = np.asarray(agent._assemble_obs(np.array([0.3], dtype=np.float32)))
            assert obs.shape == (1,)


class TestAdaptiveArms:
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
        x = agent._model.normalize_input(jnp.zeros(agent._obs_dim + 1))
        assert bool(jnp.all(jnp.isfinite(agent._model.predict_mean(x))))

    @pytest.mark.parametrize("mode", sorted(ADAPTIVE_ZONES))
    def test_offline_buffer_matches_v28_training_set(self, mode):
        agent = make_agent(ADAPTIVE_ZONES[mode], cls=AdaptivePPOPolicy)
        # visu-v28 N_train from the run's tfevents hparams.
        assert agent._offline_count == 12143
        assert agent._max_ep_len == 14
        assert agent._buffer_obs.shape[1] == agent._obs_dim

    @pytest.mark.parametrize("mode", sorted(ADAPTIVE_ZONES))
    def test_nightly_retrain_completes_and_swaps(self, mode):
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
        # plan() swallows exceptions and still returns to "idle", so only the
        # swap counter proves the retrain actually succeeded under this reward.
        assert agent._retrain_count == 1

        action, _ = agent.step(
            0.0, np.array([0.6], dtype=np.float32), plant_extra(areas0 * 1.1)
        )
        assert 0.4 <= action <= 1.3

    def test_retrain_archives_inspectable_artifacts(self, tmp_path):
        """Each completed retrain archives its policy/model/data for inspection."""
        archive = tmp_path / "retrain_archive"
        agent = make_agent(
            ADAPTIVE_ZONES["masked_log"],
            cls=AdaptivePPOPolicy,
            retrain_after_hour=0,
            retrain_archive_dir=str(archive),
            mbrl={
                "model": {"update_steps": 20},
                "model_steps_per_slice": 10,
                "ppo": {"num_envs": 64, "total_timesteps": 64 * 10 * 2},
                "chunk_updates": 1,
            },
        )
        areas0 = np.full(12, 9.0)
        agent.start(np.array([0.5], dtype=np.float32), plant_extra(areas0))
        agent.step(0.0, np.array([0.55], dtype=np.float32), plant_extra(areas0 * 1.04))
        for _ in range(100):
            agent.plan()
            if agent._phase == "idle" and agent._retrain_count == 1:
                break
        assert agent._retrain_count == 1

        runs = sorted(archive.iterdir())
        assert len(runs) == 1, f"expected one archived retrain, got {runs}"
        run = runs[0]
        assert run.name.endswith("_retrain0001")
        for sub in ("network", "obs_norm", "model"):
            assert (run / sub).is_dir(), f"missing {sub} checkpoint"

        meta = json.loads((run / "metadata.json").read_text())
        assert meta["retrain_count"] == 1
        assert meta["reward_mode"] == "masked_log"
        assert meta["obs_dim"] == 2
        assert meta["online_transitions"] == 12
        assert meta["offline_transitions"] == agent._offline_count

        online = np.load(run / "online_transitions.npz")
        assert online["obs"].shape == (12, 2)
        assert np.all(np.isfinite(online["obs"]))

        # The archived policy must be the one actually acting: restoring it
        # reproduces the live agent's parameters exactly.
        import jax
        import orbax.checkpoint as ocp
        from flax import nnx

        _, live_state = nnx.split(agent._network)
        restored = ocp.StandardCheckpointer().restore(
            str(run / "network"), target=live_state
        )
        live_leaves = jax.tree.leaves(jax.tree.map(np.asarray, live_state))
        restored_leaves = jax.tree.leaves(jax.tree.map(np.asarray, restored))
        assert len(live_leaves) == len(restored_leaves) > 0
        for live, saved in zip(live_leaves, restored_leaves, strict=True):
            np.testing.assert_array_equal(saved, live)

    def test_adaptive_configs_enable_archiving(self):
        """The deployed explore arms must archive; omitting the key disables it."""
        for zone in ADAPTIVE_ZONES.values():
            params = zone_params(zone)
            assert params["retrain_archive_dir"].startswith("/data/plant-rl/")
            assert "E21/P1" in params["retrain_archive_dir"]
        agent = make_agent(
            ADAPTIVE_ZONES["analytic"], cls=AdaptivePPOPolicy, retrain_archive_dir=None
        )
        assert agent._retrain_archive_dir is None

    def test_checkpoint_roundtrip(self, setup_checkpoint_test, tmpdir):
        def collect(agent):
            a0 = np.exp(np.linspace(-0.5, 0.5, 16))
            agent.start(np.array([0.4], dtype=np.float32), plant_extra(a0))
            agent.step(0.0, np.array([0.5], dtype=np.float32), plant_extra(a0 * 1.03))

        original, loaded = setup_checkpoint_test(
            tmpdir,
            zone_params("Z3"),
            AdaptivePPOPolicy,
            observations=(1,),
            actions=1,
            init_func=collect,
        )
        assert loaded._pointer == original._pointer == original._offline_count + 16
        assert loaded._retrain_count == original._retrain_count
        probe = jnp.asarray(np.array([0.5], dtype=np.float32))
        assert loaded._policy_action(probe) == pytest.approx(
            original._policy_action(probe), abs=1e-6
        )


class TestConfigsConsistent:
    def test_configs_match_golden_and_zone_layout(self):
        g = golden()
        expected_zone = {
            "Z1": "alliance-zone01",
            "Z2": "alliance-zone02",
            "Z3": "alliance-zone03",
            "Z4": "alliance-zone04",
        }
        for zone, (mode, head) in ZONES.items():
            cfg = json.loads((CONFIG_DIR / f"{zone}.json").read_text())
            params = cfg["metaParameters"]
            assert params["action_min"] == g["action_min"]
            assert params["action_max"] == g["action_max"]
            assert params["obs_dim"] == g[mode]["obs_dim"]
            assert params["policy"] == head.removeprefix("ppo_")
            assert params["environment"]["zone"] == expected_zone[zone]
            assert params["timezone"] == "Etc/GMT-2"
            assert params["environment"]["observation"] == "log_area"
            # Agent zones must see plants: CV pipeline stays enabled (unlike the
            # E21/P0 constant zones, which disable it deliberately).
            assert params["environment"].get("enable_cv_pipeline", True)
            # Explore arms sample; exploit arms are deterministic.
            expect_sel = "sample" if head == "ppo_explore" else "mean"
            assert params["action_selection"] == expect_sel
            # Only the explore arms retrain.
            assert ("reward_mode" in params) == (zone in ("Z3", "Z4"))
            if "reward_mode" in params:
                assert params["reward_mode"] == mode
                assert cfg["agent"].startswith("AdaptivePPOPolicy")
            else:
                assert cfg["agent"].startswith("PPOPolicy")

    def test_agent_names_resolve(self):
        from algorithms.registry import getAgent

        assert getAgent("PPOPolicy1") is PPOPolicy
        assert getAgent("PPOPolicy2") is PPOPolicy
        assert getAgent("AdaptivePPOPolicy3") is AdaptivePPOPolicy
        assert getAgent("AdaptivePPOPolicy4") is AdaptivePPOPolicy
