import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.initializers import constant, orthogonal


class Actor(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        discrete: bool = False,
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.discrete = discrete
        self.use_layer_norm = use_layer_norm
        if activation == "relu":
            self.activation = nnx.relu
        else:
            self.activation = nnx.tanh
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
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        if use_layer_norm:
            self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        if not discrete:
            self.log_std = nnx.Param(jnp.zeros(self.action_dim))

    def __call__(self, x: jax.Array):
        actor_mean = self.dense1(x)
        if self.use_layer_norm:
            actor_mean = self.ln1(actor_mean)
        actor_mean = self.activation(actor_mean)
        actor_mean = self.dense2(actor_mean)
        if self.use_layer_norm:
            actor_mean = self.ln2(actor_mean)
        actor_mean = self.activation(actor_mean)
        actor_mean = self.dense3(actor_mean)
        if self.discrete:
            return distrax.Categorical(logits=actor_mean)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.log_std.get_value())
        )


class Critic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.use_layer_norm = use_layer_norm
        if activation == "relu":
            self.activation = nnx.relu
        else:
            self.activation = nnx.tanh
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
        if use_layer_norm:
            self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        critic = self.dense1(x)
        if self.use_layer_norm:
            critic = self.ln1(critic)
        critic = self.activation(critic)
        critic = self.dense2(critic)
        if self.use_layer_norm:
            critic = self.ln2(critic)
        critic = self.activation(critic)
        critic = self.dense3(critic)
        return jnp.squeeze(critic, axis=-1)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        discrete: bool = False,
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.discrete = discrete
        self.activation = activation
        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
            activation,
            discrete,
            use_layer_norm,
            rngs=rngs,
        )
        self.critic = Critic(
            state_dim, hidden_dim, activation, use_layer_norm, rngs=rngs
        )

    def __call__(self, x):
        pi = self.actor(x)
        critic = self.critic(x)
        return pi, critic


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        zero_out_init: bool = False,
        use_layer_norm: bool = False,
    ):
        self.use_layer_norm = use_layer_norm
        self.linear1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        if use_layer_norm:
            self.ln1 = nnx.LayerNorm(hidden_features, rngs=rngs)
        if zero_out_init:
            self.linear2 = nnx.Linear(
                hidden_features,
                out_features,
                rngs=rngs,
                kernel_init=nnx.initializers.zeros,
            )
        else:
            self.linear2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        features = self.linear1(x)
        if self.use_layer_norm:
            features = self.ln1(features)
        features = nnx.tanh(features)
        y = self.linear2(features)
        return y, features


class ProjectedMLP(nnx.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, rngs: nnx.Rngs
    ):
        self.mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            rngs=rngs,
            zero_out_init=True,
            use_layer_norm=False,
        )

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        xz = jnp.concatenate([x, z], axis=-1)
        y, _ = self.mlp(xz, rngs=rngs)
        y1 = y.reshape(-1, z.shape[-1])
        return (y1 * z).sum(axis=-1)


class MLPEnsemble(nnx.Module):
    def __init__(
        self,
        num_models: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        keys = jax.random.split(rngs.params(), num_models)

        @nnx.vmap
        def create_model(key):
            return MLP(
                in_features,
                hidden_features,
                out_features,
                use_layer_norm=False,
                rngs=nnx.Rngs(params=key),
            )

        self.models = create_model(keys)

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        graphdef, states = nnx.split(self.models)

        def forward(state, inputs):
            model = nnx.merge(graphdef, state)
            y, _ = model(inputs, rngs=rngs)
            return y

        y = jax.vmap(forward, in_axes=(0, None))(states, x)
        return jnp.einsum("no,n->o", y, z)


class EpiNet(nnx.Module):
    def __init__(
        self,
        in_features: int,
        learnable_hidden_features: int,
        prior_hidden_features: int,
        base_features: int,
        out_features: int,
        index_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.learnable = ProjectedMLP(
            in_features=in_features + base_features + index_dim,
            hidden_features=learnable_hidden_features,
            out_features=out_features * index_dim,
            rngs=rngs,
        )
        self.prior = MLPEnsemble(
            num_models=index_dim,
            in_features=in_features,
            hidden_features=prior_hidden_features,
            out_features=out_features,
            rngs=rngs,
        )
        self.prior_scale = 1.0

    def __call__(self, phi, x, z, rngs: nnx.Rngs | None = None):
        return self.learnable(
            phi, z, rngs=rngs
        ) + self.prior_scale * jax.lax.stop_gradient(self.prior(x, z, rngs=rngs))


class ENN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        learnable_hidden_features: int,
        prior_hidden_features: int,
        out_features: int,
        index_dim: int,
        *,
        use_layer_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        self.base = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            use_layer_norm=use_layer_norm,
            rngs=rngs,
        )
        self.epinet = EpiNet(
            in_features=in_features,
            learnable_hidden_features=learnable_hidden_features,
            prior_hidden_features=prior_hidden_features,
            base_features=hidden_features,
            out_features=out_features,
            index_dim=index_dim,
            rngs=rngs,
        )
        self.index_dim = index_dim

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        y, features = self.base(x, rngs=rngs)
        phi = jnp.concatenate([x, features], axis=-1)
        return y, y + self.epinet(jax.lax.stop_gradient(phi), x, z, rngs=rngs)
