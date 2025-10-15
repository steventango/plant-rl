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
