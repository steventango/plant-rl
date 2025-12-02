import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn import initializers

from algorithms.nn.inac.network import network_bodies


class MLPCont(nnx.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        action_range=1.0,
        init_type="xavier",
        *,
        rngs: nnx.Rngs,
    ):
        self.body = network_bodies.FCBody(
            input_dim=obs_dim,
            hidden_units=tuple(hidden_sizes),
            init_type=init_type,
            rngs=rngs,
        )
        self.mu_layer = nnx.Linear(
            self.body.feature_dim,
            act_dim,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )
        self.log_std_logits = nnx.Param(jnp.zeros(act_dim))
        self.min_log_std = -6
        self.max_log_std = 0
        self.action_range = action_range

    def __call__(self, obs, rngs: nnx.Rngs, deterministic=False):
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = jnp.tanh(mu) * self.action_range

        log_std = jax.nn.sigmoid(self.log_std_logits.value)
        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = jnp.exp(log_std)

        pi_distribution = distrax.Normal(loc=mu, scale=std)

        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.sample(seed=rngs.sample())

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = jnp.tanh(mu) * self.action_range

        log_std = jax.nn.sigmoid(self.log_std_logits.value)
        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = jnp.exp(log_std)

        pi_distribution = distrax.Normal(loc=mu, scale=std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        return logp_pi


class MLPDirichlet(nnx.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        init_type="xavier",
        offset=0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.body = network_bodies.FCBody(
            input_dim=obs_dim,
            hidden_units=tuple(hidden_sizes),
            init_type=init_type,
            rngs=rngs,
        )
        self.alpha_layer = nnx.Linear(
            self.body.feature_dim,
            act_dim,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )
        self.act_dim = act_dim
        # Small epsilon to prevent log_prob from returning NaN at the boundaries 0 and 1
        self.epsilon = 1e-2
        self.clip_alpha = 15.0
        self.offset = offset

    def __call__(self, obs, rngs: nnx.Rngs, deterministic=False):
        alpha = self.get_alpha(obs)

        pi_distribution = distrax.Dirichlet(concentration=alpha)

        if deterministic:
            pi_action = pi_distribution.mode()
            pi_action = jnp.where(
                jnp.isnan(pi_action),
                pi_distribution.mean(),
                pi_action,
            )
        else:
            pi_action = pi_distribution.sample(seed=rngs.sample())

        clipped_action = jnp.clip(pi_action, self.epsilon, 1.0 - self.epsilon)
        clipped_action = clipped_action / jnp.sum(
            clipped_action, axis=-1, keepdims=True
        )
        logp_pi = pi_distribution.log_prob(clipped_action)

        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        alpha = self.get_alpha(obs)
        pi_distribution = distrax.Dirichlet(concentration=alpha)
        clipped_action = jnp.clip(actions, self.epsilon, 1.0 - self.epsilon)
        clipped_action = clipped_action / jnp.sum(
            clipped_action, axis=-1, keepdims=True
        )
        logp_pi = pi_distribution.log_prob(clipped_action)
        return logp_pi

    def get_alpha(self, obs):
        net_out = self.body(obs)
        alpha_logits = self.alpha_layer(net_out)
        alpha = jax.nn.sigmoid(alpha_logits) * self.clip_alpha + self.offset
        return alpha


class MLPMixtureDirichlet(nnx.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        num_components=3,
        init_type="xavier",
        offset=1.0,
        clip_alpha=15.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.body = network_bodies.FCBody(
            input_dim=obs_dim,
            hidden_units=tuple(hidden_sizes),
            init_type=init_type,
            rngs=rngs,
        )

        self.num_components = num_components
        self.act_dim = act_dim
        self.offset = offset
        self.clip_alpha = clip_alpha
        self.epsilon = 1e-2  # For numerical stability in log_prob

        self.mixture_layer = nnx.Linear(
            self.body.feature_dim,
            num_components,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )

        self.alpha_layer = nnx.Linear(
            self.body.feature_dim,
            num_components * act_dim,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, obs, rngs: nnx.Rngs, deterministic=False):
        pi_distribution = self._get_distribution(obs)

        if deterministic:
            pi_action = pi_distribution.mean()
        else:
            pi_action = pi_distribution.sample(seed=rngs.sample())

        clipped_action = jnp.clip(pi_action, self.epsilon, 1.0 - self.epsilon)
        clipped_action = clipped_action / jnp.sum(
            clipped_action, axis=-1, keepdims=True
        )

        logp_pi = pi_distribution.log_prob(clipped_action)

        return clipped_action, logp_pi

    def get_logprob(self, obs, actions):
        pi_distribution = self._get_distribution(obs)

        clipped_action = jnp.clip(actions, self.epsilon, 1.0 - self.epsilon)
        clipped_action = clipped_action / jnp.sum(
            clipped_action, axis=-1, keepdims=True
        )

        logp_pi = pi_distribution.log_prob(clipped_action)

        return logp_pi

    def _get_distribution(self, obs):
        net_out = self.body(obs)
        batch_size = obs.shape[0]

        mixture_logits = self.mixture_layer(net_out)
        alpha_logits = self.alpha_layer(net_out)

        alpha_logits = alpha_logits.reshape(
            batch_size, self.num_components, self.act_dim
        )
        alpha = jax.nn.sigmoid(alpha_logits) * self.clip_alpha + self.offset
        # alpha = jax.nn.relu(alpha_logits) + self.offset

        mixture_dist = distrax.Categorical(logits=mixture_logits)
        components_dist = distrax.Dirichlet(concentration=alpha)

        pi_distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        return pi_distribution


class MLPDiscrete(nnx.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        init_type="xavier",
        *,
        rngs: nnx.Rngs,
    ):
        self.body = network_bodies.FCBody(
            input_dim=obs_dim,
            hidden_units=tuple(hidden_sizes),
            init_type=init_type,
            rngs=rngs,
        )
        self.mu_layer = nnx.Linear(
            self.body.feature_dim,
            act_dim,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, obs, rngs: nnx.Rngs, deterministic=True):
        net_out = self.body(obs)
        logits = self.mu_layer(net_out)
        m = distrax.Categorical(logits=logits)
        action = m.sample(seed=rngs.sample())
        logp = m.log_prob(action)
        return action, logp

    def get_logprob(self, obs, actions):
        net_out = self.body(obs)
        logits = self.mu_layer(net_out)
        m = distrax.Categorical(logits=logits)
        logp_pi = m.log_prob(actions)
        return logp_pi
