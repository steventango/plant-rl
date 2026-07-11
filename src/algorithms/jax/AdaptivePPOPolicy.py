import logging
import os
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
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
from algorithms.jax.PPOPolicy import _CPU, PPOPolicy

logger = logging.getLogger("plant_rl.AdaptivePPOPolicy")


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

        mbrl_params = params.get("mbrl", {})
        self._ppo_config = derive_config(
            hparams.merged(hparams.PPO_DEFAULTS, mbrl_params.get("ppo"))
        )
        self._enn_config = hparams.merged(
            hparams.ENN_DEFAULTS, mbrl_params.get("model")
        )
        self._chunk_updates: int = mbrl_params.get("chunk_updates", 8)
        self._model_steps_per_slice: int = mbrl_params.get("model_steps_per_slice", 500)

        with jax.default_device(_CPU):
            self._load_offline_buffer(
                params["dataset_npz"], params.get("buffer_headroom", 64)
            )
            self._restore_model(params["checkpoint_path"])

        # Nightly-retrain state machine: "idle" -> "model" -> "ppo" -> "swap".
        self._phase = "idle"
        self._retrain: dict | None = None
        self._last_retrain_date = None
        self._prev_obs: np.ndarray | None = None
        self._prev_action: float | None = None

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
    def _append_transition(self, next_obs: np.ndarray) -> None:
        if self._prev_obs is None or self._prev_action is None:
            return
        i = self._pointer
        if i >= self._buffer_obs.shape[0]:
            logger.warning("online transition buffer full; dropping transition")
            return
        self._buffer_obs[i] = self._prev_obs
        self._buffer_action[i] = self._prev_action
        self._buffer_next_obs[i] = next_obs
        self._buffer_reward[i] = 0.0  # oracle reward is recomputed at retrain time
        self._buffer_terminated[i] = False
        self._buffer_truncated[i] = False
        self._pointer = i + 1

    def start(self, observation: Any, extra: dict[str, Any] | None = None):
        self._day = 0
        obs = np.asarray(self._assemble_obs(observation))
        action = self._policy_action(jnp.asarray(obs))
        self._prev_obs, self._prev_action = obs, action
        return action, {}

    def step(self, reward: float, observation: Any, extra: Any):
        self._day += 1
        obs = np.asarray(self._assemble_obs(observation))
        self._append_transition(obs)
        action = self._policy_action(jnp.asarray(obs))
        self._prev_obs, self._prev_action = obs, action
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
        env = PlantEnv(int(self._buffer_action.shape[1]), reward_mode=self._reward_mode)
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
        logger.debug(
            "Nightly retrain complete after %d PPO updates; new policy active",
            self._retrain["updates_done"],
        )
        self._reset_retrain()

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
        state["prev_obs"] = self._prev_obs
        state["prev_action"] = self._prev_action
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
        self._prev_obs = state["prev_obs"]
        self._prev_action = state["prev_action"]
