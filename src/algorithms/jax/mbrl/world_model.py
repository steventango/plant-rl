from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx


class WorldModel(nnx.Module):
    """Base class for learned dynamics models with shared normalization."""

    def __init__(
        self,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
        predict_terminated: bool | None = None,
        dyn_dim: int | None = None,
    ):
        self.obs_dim = obs_dim
        # Dynamics outputs. Defaults to obs_dim (predict the whole observation
        # delta). Set it smaller when some observation dims are deterministic and
        # supplied by the env rather than predicted — the plant day counter, for
        # instance: predicting it wastes capacity and injects a deterministic
        # quantity into the epistemic-uncertainty bonus.
        self.dyn_dim = obs_dim if dyn_dim is None else dyn_dim
        self.act_dim = act_dim
        self.eps = eps
        # Reward and termination heads are independent. Plant termination is
        # purely "day >= horizon", supplied by the oracle, so plant runs predict
        # neither; ``predict_reward_terminated`` sets both at once for the
        # historical coupled case and ``predict_terminated`` overrides the second.
        self.predict_reward = predict_reward_terminated
        self.predict_terminated = (
            predict_reward_terminated if predict_terminated is None else predict_terminated
        )
        self.input_mean = nnx.Variable(jnp.zeros(in_features))
        self.input_std = nnx.Variable(jnp.ones(in_features))
        self.delta_obs_mean = nnx.Variable(jnp.zeros(self.dyn_dim))
        self.delta_obs_std = nnx.Variable(jnp.ones(self.dyn_dim))
        self.reward_mean = nnx.Variable(jnp.zeros(()))
        self.reward_std = nnx.Variable(jnp.ones(()))

    @property
    def predict_reward_terminated(self) -> bool:
        """Legacy coupled flag: both extra heads present."""
        return self.predict_reward and self.predict_terminated

    @property
    def reward_index(self) -> int:
        """Output index of the reward head (absolute, not negative).

        Negative indices break as soon as the two extra heads are independent.
        """
        return self.dyn_dim

    @property
    def terminated_index(self) -> int:
        return self.dyn_dim + (1 if self.predict_reward else 0)

    @staticmethod
    def dyn_target(dataset) -> jax.Array:
        """Dynamics-model regression target for a batch of transitions.

        ``info["dyn_target"]`` when the loader supplied one (the plant
        differential setup makes it the de-confounded growth advantage rather
        than the raw observation delta); otherwise the observation delta.
        """
        if "dyn_target" in dataset.info:
            return dataset.info["dyn_target"]
        return dataset.info["next_obs"] - dataset.obs

    # --- Abstract per-model primitives ---

    def predict_sample(self, x, index):
        """Single input x, single index → sample output."""
        raise NotImplementedError

    def sample_index(self, key, num_samples: int):
        """Draw ``num_samples`` posterior indices from the prior."""
        raise NotImplementedError

    def predict_mean(self, x):
        """Deterministic posterior mean for a single normalized input."""
        raise NotImplementedError

    # --- Concrete uncertainty helpers ---

    def predict_samples(self, x, index):
        """x (in_features,), index (S, index_dim) → (S, out_features). Maps over S indices."""
        return jax.vmap(self.predict_sample, in_axes=(None, 0))(x, index)

    def batch_predict_sample(self, x, index):
        """x (N, in_features), index (index_dim,) → (N, out_features). Maps over N inputs with a single index."""
        return jax.vmap(self.predict_sample, in_axes=(0, None))(x, index)

    def batch_predict_samples(self, x, index):
        """x (N, in_features), index (S, index_dim) → (S, N, out_features). Maps over N inputs and S indices."""
        return jax.vmap(self.predict_samples, in_axes=(0, None), out_axes=1)(x, index)

    def variance(
        self, x: jax.Array, z: jax.Array, dims: jax.Array | None = None
    ) -> jax.Array:
        """Per-output empirical epistemic variance at a single x over S samples.

        ``dims`` restricts which output heads count. Left as None the variance
        covers every output, which is only right when every output is a genuinely
        uncertain quantity.
        """
        samples = self.predict_samples(x, z)
        if dims is not None:
            samples = samples[..., dims]
        return samples.var(axis=0)

    def uncertainty(
        self,
        x: jax.Array,
        z: jax.Array,
        kind: Literal["std", "eig"] = "std",
        reduce_output: bool = True,
        dims: jax.Array | None = None,
    ) -> jax.Array:
        """Epistemic uncertainty at a single x given S epistemic indices z."""
        var = self.variance(x, z, dims)
        if kind == "eig":
            u = 0.5 * jnp.log(1.0 + var)
        else:
            u = jnp.sqrt(var)
        return u.sum(axis=-1) if reduce_output else u

    def batch_uncertainty(
        self,
        x: jax.Array,
        z: jax.Array,
        explore_bonus: Literal["std", "eig"] = "std",
        reduce_output: bool = True,
        dims: jax.Array | None = None,
    ) -> jax.Array:
        """x (N, in_features) → (N,) or (N, out_features)."""
        return jax.vmap(
            lambda xi: self.uncertainty(xi, z, explore_bonus, reduce_output, dims)
        )(x)

    # --- Shared input/normalization helpers ---

    def encode_action(self, action):
        """action: (batch, a_dim); discrete uses a_dim=1 index column."""
        if self.act_dim is not None:
            return jax.nn.one_hot(action[:, 0], self.act_dim)
        return action

    def build_input(self, obs, action):
        """obs: (batch, obs_dim), action: (batch, a_dim)."""
        return jnp.concatenate([obs, self.encode_action(action)], axis=-1)

    def single_input(self, obs, action):
        obs = jnp.asarray(obs)
        if obs.ndim == 1:
            obs = obs[None]
        action = jnp.atleast_2d(jnp.asarray(action))
        x = self.normalize_input(self.build_input(obs, action))
        return jnp.reshape(x, (-1,))

    def update_stats(self, dataset, pointer):
        n_samples = dataset.obs.shape[0]
        mask = jnp.arange(n_samples) < pointer
        mask2d = mask[:, None]

        delta_obs = self.dyn_target(dataset)

        self.input_mean[: self.obs_dim] = jnp.mean(dataset.obs, axis=0, where=mask2d)
        self.input_std[: self.obs_dim] = jnp.maximum(
            jnp.std(dataset.obs, axis=0, where=mask2d), self.eps
        )

        if self.act_dim is None:
            self.input_mean[self.obs_dim :] = jnp.mean(
                dataset.action, axis=0, where=mask2d
            )
            self.input_std[self.obs_dim :] = jnp.maximum(
                jnp.std(dataset.action, axis=0, where=mask2d), self.eps
            )
        else:
            self.input_mean[self.obs_dim :] = 0.0
            self.input_std[self.obs_dim :] = 1.0

        self.delta_obs_mean[...] = jnp.mean(delta_obs, axis=0, where=mask2d)
        self.delta_obs_std[...] = jnp.maximum(
            jnp.std(delta_obs, axis=0, where=mask2d), self.eps
        )
        self.reward_mean[...] = jnp.mean(dataset.reward, axis=0, where=mask)
        self.reward_std[...] = jnp.maximum(
            jnp.std(dataset.reward, axis=0, where=mask), self.eps
        )

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def build_targets(self, delta_obs, reward, terminated) -> jax.Array:
        """Normalized output targets (N, out_features).

        ``delta_obs`` is the dynamics target (see :meth:`dyn_target`) — already a
        delta, not a pair of observations, because the differential setup's target
        is not any observation difference.
        """
        parts = [self.normalize_delta_obs(delta_obs)]
        if self.predict_reward:
            parts.append(self.normalize_reward(reward)[:, None])
        if self.predict_terminated:
            parts.append(terminated[:, None].astype(jnp.float32))
        return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    def normalize_delta_obs(self, delta):
        return (delta - self.delta_obs_mean) / self.delta_obs_std

    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / self.reward_std

    def denormalize_delta_obs(self, delta_norm):
        return delta_norm * self.delta_obs_std + self.delta_obs_mean

    def denormalize_reward(self, reward_norm):
        return reward_norm * self.reward_std + self.reward_mean


# --- Registry and dispatching factories ---

_REGISTRY: dict[str, dict] = {}


def register_model(name: str):
    """Decorator to register a model's build/train factories under a string key."""

    def decorator(factories: dict):
        _REGISTRY[name] = factories
        return factories

    return decorator


def make_batched_model(model_type: str, *args, **kwargs):
    return _REGISTRY[model_type]["make_batched_model"](*args, **kwargs)


def make_batched_train_model(model_type: str, *args, **kwargs):
    return _REGISTRY[model_type]["make_batched_train_model"](*args, **kwargs)


def make_batched_rngs(keys):
    """Build a seed-batched nnx.Rngs: one independent stream per seed."""

    @nnx.vmap(in_axes=0, out_axes=0)
    def build(key):
        return nnx.Rngs(key)

    return build(keys)
