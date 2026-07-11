import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from algorithms.BaseAgent import BaseAgent
from algorithms.jax.mbrl.networks import ActorCritic
from algorithms.jax.mbrl.normalization import NormalizeVecObs

# Inference (and any retraining) runs on CPU regardless of accelerator
# availability; the zone containers' GPUs are not required for these agents.
_CPU = jax.devices("cpu")[0]

# Training episodes cover day indices 0..14 (offline_data.py include_time), so
# a longer deployment episode clamps the synthesized day at the last trained one.
MAX_DAY_INDEX = 14


def restore_policy(
    checkpoint_path: str,
    policy_dir: str,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    activation: str,
    seed: int,
) -> tuple[ActorCritic, NormalizeVecObs]:
    """Restore an ActorCritic + obs normalizer saved by
    model-uncertainty-exploration's main.py (orbax StandardCheckpointer)."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    with jax.default_device(_CPU):
        network = ActorCritic(
            obs_dim,
            action_dim,
            hidden_dim,
            activation=activation,
            use_layer_norm=False,
            rngs=nnx.Rngs(seed),
        )
        obs_norm = NormalizeVecObs(jnp.zeros(obs_dim))

        network_graphdef, network_state = nnx.split(network)
        obs_norm_graphdef, obs_norm_state = nnx.split(obs_norm)

        checkpointer = ocp.StandardCheckpointer()
        network_state = checkpointer.restore(
            os.path.join(checkpoint_path, policy_dir, "network"),
            target=network_state,
        )
        obs_norm_state = checkpointer.restore(
            os.path.join(checkpoint_path, policy_dir, "obs_norm"),
            target=obs_norm_state,
        )

        network = nnx.merge(network_graphdef, network_state)
        obs_norm = nnx.merge(obs_norm_graphdef, obs_norm_state)
    return network, obs_norm


class PPOPolicy(BaseAgent):
    """Deploys a PPO policy trained offline in model-uncertainty-exploration.

    params:
        checkpoint_path: dir containing ppo_explore/ and ppo_eval/ orbax trees
        policy: "explore" or "eval" head selection
        action_selection: "mean" (deterministic) or "sample" (stochastic; the
            key is derived from (seed, local date) so a same-day container
            restart reproduces the same action)
        obs_dim: 1 for [log_area]; 2 for [log_area, day_index] where the day
            index counts 09:00 polls since episode start, clamped to
            MAX_DAY_INDEX (training episode length)
    """

    def __init__(
        self, observations: Any, actions: Any, params: dict, collector: Any, seed: int
    ):
        super().__init__(observations, actions, params, collector, seed)

        checkpoint_path: str = params["checkpoint_path"]
        policy: str = params.get("policy", "eval")  # "explore" or "eval"
        self._obs_dim: int = params.get("obs_dim", 1)
        action_dim: int = params.get("action_dim", 1)
        hidden_dim: int = params.get("hidden_dim", 64)
        activation: str = params.get("activation", "tanh")
        self._action_selection: str = params.get("action_selection", "mean")
        if self._action_selection not in ("mean", "sample"):
            raise ValueError(
                f"action_selection must be 'mean' or 'sample', "
                f"got {self._action_selection!r}"
            )
        self._tz = ZoneInfo(params.get("timezone", "Etc/GMT-2"))
        self._day = 0

        policy_dir = "ppo_explore" if policy == "explore" else "ppo_eval"
        self._network, self._obs_norm = restore_policy(
            checkpoint_path,
            policy_dir,
            self._obs_dim,
            action_dim,
            hidden_dim,
            activation,
            seed,
        )
        self._action_min: float = params["action_min"]
        self._action_max: float = params["action_max"]

    def _assemble_obs(self, observation: Any) -> jnp.ndarray:
        obs = np.asarray(self.process_observation(observation), dtype=np.float32)
        if self._obs_dim == 2 and obs.shape[0] == 1:
            day = float(min(self._day, MAX_DAY_INDEX))
            obs = np.concatenate([obs, np.asarray([day], dtype=np.float32)])
        if obs.shape[0] != self._obs_dim:
            raise ValueError(
                f"Observation has {obs.shape[0]} dims, policy expects {self._obs_dim}"
            )
        return jnp.asarray(obs)

    def _policy_action(self, obs: jnp.ndarray) -> float:
        with jax.default_device(_CPU):
            pi, _ = self._network(self._obs_norm.normalize(obs))
            if self._action_selection == "sample":
                ordinal = datetime.now(self._tz).date().toordinal()
                key = jax.random.fold_in(jax.random.PRNGKey(self.seed), ordinal)
                action = pi.sample(seed=key)
            else:
                action = pi.mean()
        return float(np.clip(np.asarray(action)[0], self._action_min, self._action_max))

    def _select_action(self, observation: Any) -> float:
        return self._policy_action(self._assemble_obs(observation))

    def start(self, observation: Any, extra: dict[str, Any] | None = None):
        self._day = 0
        return self._select_action(observation), {}

    def step(self, reward: float, observation: Any, extra: Any):
        self._day += 1
        return self._select_action(observation), {}

    def end(self, reward: float, extra: Any) -> dict:
        return {}

    def plan(self) -> None:
        pass

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        state = super().__getstate__()
        state["day"] = self._day
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._day = state.get("day", 0)
