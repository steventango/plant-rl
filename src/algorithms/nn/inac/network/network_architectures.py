import jax.numpy as jnp
from flax import nnx
from jax.nn import initializers

from algorithms.nn.inac.network import network_bodies


class FCNetwork(nnx.Module):
    def __init__(
        self,
        input_units,
        hidden_units,
        output_units,
        head_activation=lambda x: x,
        *,
        rngs: nnx.Rngs,
    ):
        self.body = network_bodies.FCBody(
            input_dim=input_units,
            hidden_units=tuple(hidden_units),
            init_type='xavier',
            rngs=rngs,
        )
        self.fc_head = nnx.Linear(
            self.body.feature_dim,
            output_units,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )
        self.head_activation = head_activation

    def __call__(self, x):
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))
        y = self.body(x)
        y = self.fc_head(y)
        y = self.head_activation(y)
        return y


class DoubleCriticDiscrete(nnx.Module):
    def __init__(
        self,
        input_units,
        hidden_units,
        output_units,
        *,
        rngs: nnx.Rngs,
    ):
        self.q1_net = FCNetwork(
            input_units, hidden_units, output_units, rngs=rngs
        )
        self.q2_net = FCNetwork(
            input_units, hidden_units, output_units, rngs=rngs
        )

    def __call__(self, x, a):
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        q1 = jnp.take_along_axis(q1, a[:, None], axis=1).squeeze(axis=1)
        q2 = jnp.take_along_axis(q2, a[:, None], axis=1).squeeze(axis=1)
        q_pi = jnp.minimum(q1, q2)
        return q_pi, q1, q2


class DoubleCriticNetwork(nnx.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_units,
        *,
        rngs: nnx.Rngs,
    ):
        # Q1 architecture
        self.body1 = network_bodies.FCBody(
            input_dim=num_inputs + num_actions,
            hidden_units=tuple(hidden_units),
            rngs=rngs,
        )
        self.head1 = nnx.Linear(
            self.body1.feature_dim,
            1,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )
        # Q2 architecture
        self.body2 = network_bodies.FCBody(
            input_dim=num_inputs + num_actions,
            hidden_units=tuple(hidden_units),
            rngs=rngs,
        )
        self.head2 = nnx.Linear(
            self.body2.feature_dim,
            1,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, state, action):
        if len(state.shape) > 2:
            state = state.reshape((state.shape[0], -1))
            action = action.reshape((action.shape[0], -1))
        elif len(state.shape) == 1:
            state = state.reshape((1, -1))
            action = action.reshape((1, -1))

        xu = jnp.concatenate([state, action], axis=1)

        q1 = self.head1(self.body1(xu))
        q2 = self.head2(self.body2(xu))

        q_pi = jnp.minimum(q1, q2)
        return q_pi.squeeze(-1), q1.squeeze(-1), q2.squeeze(-1)
