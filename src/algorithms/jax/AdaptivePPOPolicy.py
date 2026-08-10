import json
import logging
import os
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
from flax import nnx

from algorithms.jax.mbrl import hparams
from algorithms.jax.mbrl.enn import _make_batched_model, _train_model
from algorithms.jax.mbrl.model_env import ModelEnvironment, reset_weights
from algorithms.jax.mbrl.plant_env import PlantEnv, PlantEnvParams, RewardMode
from algorithms.jax.mbrl.ppo import (
    Transition,
    derive_config,
    make_train_chunk,
    make_train_init,
    make_train_state,
)
from algorithms.jax.mbrl.wrappers import ClipAction, LogWrapper, VecEnv
from algorithms.jax.PPOPolicy import _CPU, MAX_DAY_INDEX, PPOPolicy

logger = logging.getLogger("plant_rl.AdaptivePPOPolicy")

# Per-plant reward (Δ log-area) outlier band, matching plant-data's offline
# transform_outlier_detection(q1=0.01, q2=0.99).
_OUTLIER_Q_LOW = 0.01
_OUTLIER_Q_HIGH = 0.99
# Minimum plants in a day's batch before the quantile trim is meaningful.
_MIN_PLANTS_FOR_IQR = 8


def _unstack(batched, i):
    graphdef, state = nnx.split(batched)
    return nnx.merge(graphdef, jax.tree.map(lambda x: x[i], state))


def _numpy_state(module) -> Any:
    return jax.tree.map(np.asarray, nnx.state(module))


def _load_state(module, state) -> None:
    nnx.update(module, jax.tree.map(jnp.asarray, state))


class AdaptivePPOPolicy(PPOPolicy):
    """PPOPolicy that updates its world model and retrains its policy nightly.

    Mirrors the offline plan-every loop of model-uncertainty-exploration:
    every night the ENN dynamics model is warm-started and updated on the
    offline dataset plus the transitions collected so far, then a fresh PPO
    explore policy (alpha=0, beta=1) is trained from scratch inside the model
    environment with the oracle plant reward, and swapped in as the acting
    policy.

    The whole retrain runs on CPU inside the RlGlue plan() hook, which holds a
    lock shared with step(). plan() therefore works in bounded slices — each
    call does at most one model-update or PPO chunk and returns — so a 09:00
    poll is never delayed by more than one slice. A hard deadline aborts any
    cycle still running in the morning.

    Extra params (beyond PPOPolicy):
        reward_mode: "analytic" | "masked" | ... — oracle reward for retraining
        dataset_npz: offline replay exported by
            model-uncertainty-exploration/scripts/export_offline_transitions.py
        retrain_after_hour: local hour after which the nightly cycle may start
        retrain_deadline_hour: local morning hour at which an unfinished cycle
            is aborted (old policy kept)
        buffer_headroom: online-transition capacity on top of the offline data
        mbrl: optional dict with "ppo" / "model" hparam overrides plus
            "chunk_updates" (PPO updates per plan() slice) and
            "model_steps_per_slice" (ENN SGD steps per plan() slice)
    """

    def __init__(
        self, observations: Any, actions: Any, params: dict, collector: Any, seed: int
    ):
        super().__init__(observations, actions, params, collector, seed)

        self._reward_mode: RewardMode = params["reward_mode"]
        self._retrain_after_hour: int = params.get("retrain_after_hour", 21)
        self._retrain_deadline_hour: int = params.get("retrain_deadline_hour", 8)
        # Where to archive each night's retrain artifacts for later inspection;
        # None disables archiving.
        self._retrain_archive_dir: str | None = params.get("retrain_archive_dir")

        mbrl_params = params.get("mbrl", {})
        self._ppo_config = derive_config(
            hparams.merged(hparams.PPO_DEFAULTS, mbrl_params.get("ppo"))
        )
        self._enn_config = hparams.merged(
            hparams.ENN_DEFAULTS, mbrl_params.get("model")
        )
        self._chunk_updates: int = mbrl_params.get("chunk_updates", 8)
        self._model_steps_per_slice: int = mbrl_params.get("model_steps_per_slice", 500)

        # Online transitions are per-plant (~one per pot per daily poll), so a
        # whole experiment adds days * plants rows on top of the offline data.
        with jax.default_device(_CPU):
            self._load_offline_buffer(
                params["dataset_npz"], params.get("buffer_headroom", 4096)
            )
            self._restore_model(params["checkpoint_path"])

        # Nightly-retrain state machine: "idle" -> "model" -> "ppo" -> "swap".
        self._phase = "idle"
        self._retrain: dict | None = None
        self._last_retrain_date = None
        # Completed retrains that actually swapped in a new policy. A failed
        # cycle is swallowed by plan()'s except and still returns to "idle", so
        # this is the only positive evidence that a retrain succeeded.
        self._retrain_count = 0
        # Per-pot log clean-area from the previous poll, for day-to-day matching.
        self._prev_plants: dict | None = None
        self._prev_action: float | None = None
        self._prev_day: int = 0

    # ------------------
    # -- Construction --
    # ------------------
    def _load_offline_buffer(self, npz_path: str, headroom: int) -> None:
        npz = np.load(os.path.abspath(npz_path))
        n = int(npz["obs"].shape[0])
        obs_dim = int(npz["obs"].shape[1])
        if obs_dim != self._obs_dim:
            raise ValueError(
                f"dataset_npz obs_dim {obs_dim} != configured obs_dim {self._obs_dim}"
            )
        capacity = n + headroom

        def alloc(name, extra_dims=()):
            arr = np.zeros((capacity, *extra_dims), dtype=npz[name].dtype)
            arr[:n] = npz[name]
            return arr

        self._buffer_obs = alloc("obs", (obs_dim,))
        self._buffer_action = alloc("action", (npz["action"].shape[1],))
        self._buffer_reward = alloc("reward")
        self._buffer_next_obs = alloc("next_obs", (obs_dim,))
        self._buffer_terminated = alloc("terminated")
        self._buffer_truncated = alloc("truncated")
        self._offline_count = n
        self._pointer = n

        self._area_min = float(npz["area_min"])
        self._area_max = float(npz["area_max"])
        self._act_low = np.asarray(npz["act_low"], dtype=np.float32)
        self._act_high = np.asarray(npz["act_high"], dtype=np.float32)
        self._max_ep_len = int(npz["max_ep_len"])

    def _restore_model(self, checkpoint_path: str) -> None:
        action_dim = int(self._buffer_action.shape[1])
        in_features = self._obs_dim + action_dim
        keys = jax.random.split(jax.random.PRNGKey(self.seed), 1)
        models, _ = _make_batched_model(
            self._enn_config,
            in_features,
            self._obs_dim,
            self._obs_dim,
            None,
            keys,
            predict_reward_terminated=False,
        )
        model = _unstack(models, 0)
        graphdef, state = nnx.split(model)
        state = ocp.StandardCheckpointer().restore(
            os.path.join(os.path.abspath(checkpoint_path), "model"), target=state
        )
        self._model = nnx.merge(graphdef, state)

    def _dataset(self) -> Transition:
        zeros = jnp.zeros(self._buffer_obs.shape[0])
        return Transition(
            terminated=jnp.asarray(self._buffer_terminated),
            truncated=jnp.asarray(self._buffer_truncated),
            action=jnp.asarray(self._buffer_action),
            value=zeros,
            next_value=zeros,
            reward=jnp.asarray(self._buffer_reward),
            log_prob=zeros,
            obs=jnp.asarray(self._buffer_obs),
            info={"next_obs": jnp.asarray(self._buffer_next_obs)},
        )

    # -------------------------
    # -- Acting / transitions --
    # -------------------------
    def _plant_log_areas(self, extra: Any) -> dict:
        """``{pot_id: log(clean_area)}`` for valid plants in ``extra['df']``.

        The world model is trained on individual per-plant transitions, so each
        daily poll should yield one transition per plant, not a single zone
        aggregate. Dead plants (``clean_area <= 0``) and CV failures (NaN) are
        dropped. Plants are keyed on ``pot_id`` (the stable physical pot present
        in both the live CV output and the replay data), falling back to
        ``plant_id`` then row index.
        """
        df = extra.get("df") if isinstance(extra, dict) else None
        if df is None or len(df) == 0 or "clean_area" not in df.columns:
            return {}
        id_col = next((c for c in ("pot_id", "plant_id") if c in df.columns), None)
        areas = np.asarray(
            pd.to_numeric(df["clean_area"], errors="coerce"), dtype=float
        )
        ids = np.asarray(df[id_col]) if id_col is not None else np.arange(len(areas))
        return {
            pid: float(np.log(a))
            for pid, a in zip(ids, areas, strict=False)
            if np.isfinite(a) and a > 0.0
        }

    def _obs_vector(self, log_area: float, day: int) -> np.ndarray:
        if self._obs_dim == 2:
            return np.array(
                [log_area, float(min(day, MAX_DAY_INDEX))], dtype=np.float32
            )
        return np.array([log_area], dtype=np.float32)

    def _append_row(self, obs_vec: np.ndarray, next_vec: np.ndarray) -> bool:
        i = self._pointer
        if i >= self._buffer_obs.shape[0]:
            logger.warning("online transition buffer full; dropping transition")
            return False
        self._buffer_obs[i] = obs_vec
        self._buffer_action[i] = self._prev_action
        self._buffer_next_obs[i] = next_vec
        self._buffer_reward[i] = 0.0  # oracle recomputes reward at retrain time
        self._buffer_terminated[i] = False
        self._buffer_truncated[i] = False
        self._pointer = i + 1
        return True

    def _collect_transitions(self, curr_plants: dict) -> int:
        """Append one transition per pot seen on both the previous and this poll.

        Mirrors the offline per-plant dataset: matches plants day-to-day by pot
        id and, like plant-data's transform_outlier_detection, drops per-plant
        Δlog-area (reward) outside the [1%, 99%] quantiles of the day's batch.
        """
        if self._prev_plants is None or self._prev_action is None:
            return 0
        # When the CV pipeline fails, PlantGrowthChamber._reuse_last_df hands the
        # previous frame back verbatim so the policy keeps acting on a plausible
        # area. That is right for acting but wrong for learning: every pot would
        # yield an exactly-zero Δ log-area, teaching the world model a growth
        # response that was never observed. Real CV output never repeats
        # bit-for-bit, so identical readings mean a reused frame — skip it.
        if curr_plants == self._prev_plants:
            logger.warning(
                "Plant readings identical to the previous poll (reused CV frame); "
                "skipping %d fabricated zero-growth transitions",
                len(curr_plants),
            )
            return 0
        shared = [pid for pid in curr_plants if pid in self._prev_plants]
        if not shared:
            return 0
        prev = np.array([self._prev_plants[pid] for pid in shared], dtype=np.float32)
        nxt = np.array([curr_plants[pid] for pid in shared], dtype=np.float32)
        delta = nxt - prev  # per-plant reward (Δ log clean-area)
        keep = np.isfinite(delta)
        if int(keep.sum()) >= _MIN_PLANTS_FOR_IQR:
            valid = delta[keep]
            lo = np.quantile(valid, _OUTLIER_Q_LOW)
            hi = np.quantile(valid, _OUTLIER_Q_HIGH)
            keep &= (delta >= lo) & (delta <= hi)
        count = 0
        for j in np.nonzero(keep)[0]:
            obs_vec = self._obs_vector(float(prev[j]), self._prev_day)
            next_vec = self._obs_vector(float(nxt[j]), self._day)
            count += int(self._append_row(obs_vec, next_vec))
        return count

    def start(self, observation: Any, extra: dict[str, Any] | None = None):
        self._day = 0
        obs = np.asarray(self._assemble_obs(observation))
        action = self._policy_action(jnp.asarray(obs))
        self._prev_plants = self._plant_log_areas(extra)
        self._prev_action = action
        self._prev_day = self._day
        return action, {}

    def step(self, reward: float, observation: Any, extra: Any):
        self._day += 1
        obs = np.asarray(self._assemble_obs(observation))
        curr_plants = self._plant_log_areas(extra)
        self._collect_transitions(curr_plants)
        action = self._policy_action(jnp.asarray(obs))
        self._prev_plants = curr_plants
        self._prev_action = action
        self._prev_day = self._day
        return action, {}

    # ---------------------------------
    # -- Nightly retrain via plan() --
    # ---------------------------------
    def plan(self) -> None:
        try:
            now = datetime.now(self._tz)
            if self._phase == "idle":
                if (
                    now.hour < self._retrain_after_hour
                    or self._last_retrain_date == now.date()
                ):
                    return
                # Mark the night as consumed up front: even a failed or aborted
                # cycle must not spin retries all night.
                self._last_retrain_date = now.date()
                self._setup_retrain(now)
                return
            if self._retrain_deadline_hour <= now.hour < self._retrain_after_hour:
                logger.warning(
                    "Nightly retrain hit the %02d:00 deadline in phase %r; "
                    "keeping the current policy",
                    self._retrain_deadline_hour,
                    self._phase,
                )
                self._reset_retrain()
                return
            with jax.default_device(_CPU):
                if self._phase == "model":
                    self._model_slice()
                elif self._phase == "ppo":
                    self._ppo_slice()
                elif self._phase == "swap":
                    self._swap()
        except Exception:
            logger.exception("Nightly retrain failed; keeping the current policy")
            self._reset_retrain()

    def _reset_retrain(self) -> None:
        self._phase = "idle"
        self._retrain = None

    def _setup_retrain(self, now: datetime) -> None:
        logger.debug(
            "Starting nightly retrain (%s, %d transitions: %d offline + %d online)",
            self._reward_mode,
            self._pointer,
            self._offline_count,
            self._pointer - self._offline_count,
        )
        with jax.default_device(_CPU):
            base_key = jax.random.fold_in(
                jax.random.PRNGKey(self.seed), now.date().toordinal()
            )
            model_key, ppo_key, train_key = jax.random.split(base_key, 3)

            tx = optax.adamw(self._enn_config["LR"], weight_decay=1e-4)
            not_prior = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))
            optimizer = nnx.Optimizer(self._model, tx, wrt=not_prior)
            metrics = nnx.MultiMetric(
                loss=nnx.metrics.Average("loss"),
                delta_next_state_loss=nnx.metrics.Average("delta_next_state_loss"),
                reward_loss=nnx.metrics.Average("reward_loss"),
                terminated_loss=nnx.metrics.Average("terminated_loss"),
            )
            self._retrain = {
                "dataset": self._dataset(),
                "optimizer": optimizer,
                "metrics": metrics,
                "rngs": nnx.Rngs(model_key),
                "model_steps_done": 0,
                "ppo_key": ppo_key,
                "train_key": train_key,
                "updates_done": 0,
            }
        self._phase = "model"

    def _model_slice(self) -> None:
        assert self._retrain is not None
        r = self._retrain
        total = int(self._enn_config["UPDATE_STEPS"])
        steps = min(self._model_steps_per_slice, total - r["model_steps_done"])
        minibatch_size = min(self._pointer, int(self._enn_config["MINIBATCH_SIZE"]))
        history = _train_model(
            self._model,
            r["optimizer"],
            r["metrics"],
            r["dataset"],
            steps,
            self._pointer,
            minibatch_size,
            r["rngs"],
        )
        r["model_steps_done"] += steps
        if r["model_steps_done"] >= total:
            loss = float(np.asarray(history["loss"])[-1])
            logger.debug("Model update done (%d steps, loss %.5f)", total, loss)
            self._setup_ppo()

    def _setup_ppo(self) -> None:
        assert self._retrain is not None
        r = self._retrain
        # include_time keyed off the agent's own obs_dim so the retrain env's
        # observation always matches the deployed policy's input (masked modes
        # force time on regardless; non-masked modes may still opt in).
        env = PlantEnv(
            int(self._buffer_action.shape[1]),
            reward_mode=self._reward_mode,
            include_time=self._obs_dim == 2,
        )
        if env.obs_dim != self._obs_dim:
            raise ValueError(
                f"retrain env obs_dim {env.obs_dim} != agent obs_dim {self._obs_dim}"
            )
        env_params = PlantEnvParams(
            area_min=self._area_min,
            area_max=self._area_max,
            act_low=jnp.asarray(self._act_low),
            act_high=jnp.asarray(self._act_high),
            max_steps_in_episode=self._max_ep_len,
        )
        model_env = VecEnv(
            ClipAction(
                LogWrapper(
                    ModelEnvironment(
                        env,
                        env_params,  # pyright: ignore[reportArgumentType]
                        prediction_mode="sample",
                        explore_bonus="eig",
                        oracle_reward_terminated=True,
                        reset_source="init",
                        max_steps_in_episode=self._max_ep_len,
                    )
                )
            )
        )
        dataset = r["dataset"]
        model_env_params = model_env.default_params.replace(
            model=self._model,
            alpha=hparams.MODEL_ENV_DEFAULTS["ALPHA"],
            beta=hparams.MODEL_ENV_DEFAULTS["BETA"],
            init_obs=dataset.obs,
            init_weights=reset_weights(
                dataset.terminated, dataset.truncated, self._pointer, "init"
            ),
        )
        train_state = make_train_state(
            self._ppo_config, model_env, model_env_params, nnx.Rngs(params=r["ppo_key"])
        )
        runner_state = make_train_init(model_env, self._ppo_config)(
            train_state, model_env_params, r["train_key"]
        )
        r["model_env_params"] = model_env_params
        r["runner_state"] = runner_state
        r["train_chunk"] = nnx.jit(
            make_train_chunk(model_env, self._ppo_config, self._chunk_updates)
        )
        self._phase = "ppo"

    def _ppo_slice(self) -> None:
        assert self._retrain is not None
        r = self._retrain
        r["runner_state"], _ = r["train_chunk"](
            r["runner_state"], r["model_env_params"]
        )
        r["updates_done"] += self._chunk_updates
        if r["updates_done"] >= self._ppo_config["NUM_UPDATES"]:
            self._phase = "swap"

    def _swap(self) -> None:
        assert self._retrain is not None
        network, _, obs_norm, _ = self._retrain["runner_state"][0]
        self._network = network
        self._obs_norm = obs_norm
        self._retrain_count += 1
        self._archive_retrain()
        logger.debug(
            "Nightly retrain complete after %d PPO updates; new policy active "
            "(completed retrains: %d)",
            self._retrain["updates_done"],
            self._retrain_count,
        )
        self._reset_retrain()

    def _archive_retrain(self) -> None:
        """Archive this night's retrain artifacts for offline inspection.

        Writes the swapped-in policy, its obs normalizer and the updated world
        model as orbax checkpoints — the same layout main.py saves, so the
        training repo's tooling can load them directly — alongside the online
        transitions that drove the retrain and a metadata JSON. Best-effort: a
        failure here must never cost us the (already completed) retrain.
        """
        if self._retrain_archive_dir is None:
            return
        assert self._retrain is not None
        try:
            stamp = (
                self._last_retrain_date.isoformat()
                if self._last_retrain_date is not None
                else "unknown-date"
            )
            # Never write into an existing archive: orbax refuses to overwrite,
            # which would lose this night's artifacts entirely. Suffix instead so
            # a stale directory (e.g. left by a test run, or a restart that reset
            # the counter) can't cost us a real archive.
            root = os.path.abspath(self._retrain_archive_dir)
            base = os.path.join(root, f"{stamp}_retrain{self._retrain_count:04d}")
            out = base
            for suffix in range(1, 100):
                if not os.path.exists(out):
                    break
                out = f"{base}_{suffix}"
            else:
                raise RuntimeError(f"no free archive directory under {base}")
            if out != base:
                logger.warning(
                    "Archive %s already existed; writing to %s instead", base, out
                )
            os.makedirs(out, exist_ok=True)

            checkpointer = ocp.StandardCheckpointer()
            for name, module in (
                ("network", self._network),
                ("obs_norm", self._obs_norm),
                ("model", self._model),
            ):
                _, state = nnx.split(module)
                checkpointer.save(os.path.join(out, name), state)
            checkpointer.wait_until_finished()

            n, p = self._offline_count, self._pointer
            np.savez(
                os.path.join(out, "online_transitions.npz"),
                obs=self._buffer_obs[n:p],
                action=self._buffer_action[n:p],
                next_obs=self._buffer_next_obs[n:p],
            )
            with open(os.path.join(out, "metadata.json"), "w") as f:
                json.dump(
                    {
                        "retrain_count": self._retrain_count,
                        "date": stamp,
                        "reward_mode": self._reward_mode,
                        "obs_dim": self._obs_dim,
                        "offline_transitions": n,
                        "online_transitions": p - n,
                        "ppo_updates": self._retrain["updates_done"],
                        "model_update_steps": self._retrain["model_steps_done"],
                        "policy_head": "ppo_explore (retrained)",
                    },
                    f,
                    indent=1,
                )
            logger.debug("Archived retrain artifacts to %s", out)
        except Exception:
            logger.exception("Failed to archive retrain artifacts (policy is fine)")

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        state = super().__getstate__()
        n, p = self._offline_count, self._pointer
        state["online_transitions"] = {
            "obs": self._buffer_obs[n:p].copy(),
            "action": self._buffer_action[n:p].copy(),
            "next_obs": self._buffer_next_obs[n:p].copy(),
        }
        state["network_state"] = _numpy_state(self._network)
        state["obs_norm_state"] = _numpy_state(self._obs_norm)
        state["model_state"] = _numpy_state(self._model)
        state["last_retrain_date"] = self._last_retrain_date
        state["retrain_count"] = self._retrain_count
        state["prev_plants"] = self._prev_plants
        state["prev_action"] = self._prev_action
        state["prev_day"] = self._prev_day
        return state

    def __setstate__(self, state):
        # Re-inits from the frozen checkpoint + npz, then overwrites with the
        # latest retrained parameters and re-appends the collected transitions.
        # An in-flight retrain cycle is dropped (phase resets to "idle"); the
        # consumed last_retrain_date means it retries the next night.
        super().__setstate__(state)
        with jax.default_device(_CPU):
            _load_state(self._network, state["network_state"])
            _load_state(self._obs_norm, state["obs_norm_state"])
            _load_state(self._model, state["model_state"])
        online = state["online_transitions"]
        count = online["obs"].shape[0]
        n = self._offline_count
        self._buffer_obs[n : n + count] = online["obs"]
        self._buffer_action[n : n + count] = online["action"]
        self._buffer_next_obs[n : n + count] = online["next_obs"]
        self._pointer = n + count
        self._last_retrain_date = state["last_retrain_date"]
        self._retrain_count = state.get("retrain_count", 0)
        self._prev_plants = state.get("prev_plants")
        self._prev_action = state["prev_action"]
        self._prev_day = state.get("prev_day", 0)
