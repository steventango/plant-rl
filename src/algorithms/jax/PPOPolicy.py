import os
from typing import Any

import distrax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.nn.initializers import constant, orthogonal

from algorithms.BaseAgent import BaseAgent


# ── Network architecture (must match model-uncertainty-exploration/networks.py) ──


class _Actor(nnx.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, activation="tanh", *, rngs):
        self.action_dim = action_dim
        self.activation = nnx.tanh if activation != "relu" else nnx.relu
        self.dense1 = nnx.Linear(
            state_dim,
            hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            hidden_dim,
            hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense3 = nnx.Linear(
            hidden_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        mean = self.dense3(x)
        return distrax.MultivariateNormalDiag(mean, jnp.exp(self.log_std.get_value()))


class _Critic(nnx.Module):
    def __init__(self, state_dim, hidden_dim, activation="tanh", *, rngs):
        self.activation = nnx.tanh if activation != "relu" else nnx.relu
        self.dense1 = nnx.Linear(
            state_dim,
            hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            hidden_dim,
            hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense3 = nnx.Linear(
            hidden_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        return jnp.squeeze(self.dense3(x), axis=-1)


class _ActorCritic(nnx.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, activation="tanh", *, rngs):
        self.actor = _Actor(state_dim, action_dim, hidden_dim, activation, rngs=rngs)
        self.critic = _Critic(state_dim, hidden_dim, activation, rngs=rngs)

    def __call__(self, x):
        return self.actor(x), self.critic(x)


class _NormalizeVecObs(nnx.Module):
    def __init__(self, obs_dim, eps=1e-8):
        self.eps = eps
        self.mean = nnx.Variable(jnp.zeros(obs_dim))
        self.var = nnx.Variable(jnp.ones(obs_dim))
        self.count = nnx.Variable(jnp.array(1e-4))

    def normalize(self, x):
        return (x - self.mean.value) / jnp.sqrt(self.var.value + self.eps)


# ── Agent ────────────────────────────────────────────────────────────────────


class PPOPolicy(BaseAgent):
    def __init__(
        self, observations: Any, actions: Any, params: dict, collector: Any, seed: int
    ):
        super().__init__(observations, actions, params, collector, seed)

        checkpoint_path: str = params["checkpoint_path"]
        policy: str = params.get("policy", "eval")  # "explore" or "eval"
        obs_dim: int = params.get("obs_dim", 1)
        action_dim: int = params.get("action_dim", 1)
        hidden_dim: int = params.get("hidden_dim", 64)
        activation: str = params.get("activation", "tanh")

        rngs = nnx.Rngs(seed)
        network = _ActorCritic(obs_dim, action_dim, hidden_dim, activation, rngs=rngs)
        obs_norm = _NormalizeVecObs(obs_dim)

        network_graphdef, network_state = nnx.split(network)
        obs_norm_graphdef, obs_norm_state = nnx.split(obs_norm)

        policy_dir = "ppo_explore" if policy == "explore" else "ppo_eval"
        checkpointer = ocp.StandardCheckpointer()
        network_state = checkpointer.restore(
            os.path.join(checkpoint_path, policy_dir, "network"),
            target=network_state,
        )
        obs_norm_state = checkpointer.restore(
            os.path.join(checkpoint_path, policy_dir, "obs_norm"),
            target=obs_norm_state,
        )

        self._network = nnx.merge(network_graphdef, network_state)
        self._obs_norm = nnx.merge(obs_norm_graphdef, obs_norm_state)
        self._action_min: float = params["action_min"]
        self._action_max: float = params["action_max"]

    def _select_action(self, observation: Any) -> float:
        obs = jnp.asarray(self.process_observation(observation), dtype=jnp.float32)
        obs_normalized = self._obs_norm.normalize(obs)
        pi, _ = self._network(obs_normalized)
        return float(
            np.clip(np.asarray(pi.mean())[0], self._action_min, self._action_max)
        )

    def start(self, observation: Any, extra: Any):
        return self._select_action(observation), {}

    def step(self, reward: float, observation: Any, extra: Any):
        return self._select_action(observation), {}

    def end(self, reward: float, extra: Any) -> dict:
        return {}

    def plan(self) -> None:
        pass
