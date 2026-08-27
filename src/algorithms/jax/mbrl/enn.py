import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from flax import nnx
import optax

from algorithms.jax.mbrl.networks import ENN
from algorithms.jax.mbrl.world_model import WorldModel, register_model


class ENNModel(WorldModel):
    """ENN-based dynamics model. Epistemic uncertainty via index-variable sampling."""

    def __init__(
        self,
        enn: ENN,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
        predict_terminated: bool | None = None,
        dyn_dim: int | None = None,
    ):
        super().__init__(
            in_features=in_features,
            obs_dim=obs_dim,
            act_dim=act_dim,
            eps=eps,
            predict_reward_terminated=predict_reward_terminated,
            predict_terminated=predict_terminated,
            dyn_dim=dyn_dim,
        )
        self.enn = enn

    @property
    def index_dim(self):
        return self.enn.index_dim

    def predict_sample(self, x, index):
        """Single input x, single index vector → sample output."""
        return self.enn(x, index)[1]

    def sample_index(self, key, num_samples: int):
        return jax.random.normal(key, (num_samples, self.index_dim))

    def predict_mean(self, x):
        """Deterministic base-network output for a single normalized input."""
        y, _ = self.enn.base(x)
        return y


def _loss_fn(model: ENNModel, batch, rngs: nnx.Rngs):
    sigma = 1.0
    x = model.build_input(batch.obs, batch.action)
    x = model.normalize_input(x)
    z = jax.random.normal(rngs(), shape=(model.index_dim,))
    logits = model.batch_predict_sample(x, z)

    delta_next_state_c = jax.random.normal(
        rngs(), shape=(batch.obs.shape[0], model.index_dim)
    )
    delta_next_state_c = delta_next_state_c / jnp.linalg.norm(
        delta_next_state_c, axis=-1, keepdims=True
    )

    delta_next_state = model.normalize_delta_obs(model.dyn_target(batch))
    delta_next_state_target = delta_next_state + sigma * (delta_next_state_c * z).sum(
        axis=-1, keepdims=True
    )
    delta_next_state_loss = (
        logits[..., : model.dyn_dim] - delta_next_state_target
    ) ** 2
    delta_next_state_loss = delta_next_state_loss.mean()

    if model.predict_reward:
        reward_c = jax.random.normal(
            rngs(), shape=(batch.obs.shape[0], model.index_dim)
        )
        reward_c = reward_c / jnp.linalg.norm(reward_c, axis=-1, keepdims=True)
        reward_target = model.normalize_reward(batch.reward) + sigma * (
            reward_c * z
        ).sum(axis=-1)
        reward_loss = (logits[..., model.reward_index] - reward_target) ** 2
        reward_loss = reward_loss.mean()
    else:
        reward_loss = jnp.zeros(())

    if model.predict_terminated:
        terminated_c = jax.random.normal(
            rngs(), shape=(batch.obs.shape[0], model.index_dim)
        )
        terminated_c = terminated_c / jnp.linalg.norm(
            terminated_c, axis=-1, keepdims=True
        )
        p = 0.5
        mask = ((terminated_c * z).sum(axis=-1) > norm.ppf(p)).astype(jnp.float32)
        terminated_target = batch.terminated.astype(jnp.float32)
        terminated_pred = logits[..., model.terminated_index]
        terminated_loss = optax.sigmoid_binary_cross_entropy(
            terminated_pred, terminated_target
        )
        terminated_loss = (terminated_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        terminated_loss = jnp.zeros(())

    loss = delta_next_state_loss + reward_loss + terminated_loss
    return loss, (delta_next_state_loss, reward_loss, terminated_loss)


@nnx.jit
def _train_step(
    model: ENNModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, batch, rngs)
    delta_next_state_loss, reward_loss, terminated_loss = aux
    metrics.update(
        loss=loss,
        delta_next_state_loss=delta_next_state_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
    )
    optimizer.update(model, grads)


@nnx.jit
def _eval_step(
    model: ENNModel,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    loss, aux = _loss_fn(model, batch, rngs)
    delta_next_state_loss, reward_loss, terminated_loss = aux
    metrics.update(
        loss=loss,
        delta_next_state_loss=delta_next_state_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
    )


def _train_model(
    model: ENNModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    dataset,
    update_steps: int,
    pointer: int,
    minibatch_size: int,
    rngs: nnx.Rngs,
):
    model.update_stats(dataset, pointer)

    def train_step_fn(train_state, _):
        model, optimizer, metrics, rngs = train_state
        indices = jax.random.randint(rngs(), (minibatch_size,), 0, pointer)
        minibatch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices, axis=0), dataset
        )
        metrics.reset()
        _train_step(model, optimizer, metrics, rngs, minibatch)
        return (model, optimizer, metrics, rngs), metrics.compute()

    train_state = (model, optimizer, metrics, rngs)
    _, history = nnx.scan(train_step_fn, length=update_steps)(train_state, None)
    return history


def _make_batched_model(
    model_config,
    in_features,
    obs_dim,
    out_features,
    act_dim,
    keys,
    predict_reward_terminated: bool = True,
    predict_terminated: bool | None = None,
    dyn_dim: int | None = None,
):
    @nnx.vmap
    def build(key):
        enn = ENN(
            in_features,
            model_config["HIDDEN_DIM"],
            model_config["LEARNABLE_HIDDEN_DIM"],
            model_config["PRIOR_HIDDEN_DIM"],
            out_features,
            model_config["INDEX_DIM"],
            use_layer_norm=model_config.get("USE_LAYER_NORM", False),
            rngs=nnx.Rngs(params=key),
        )
        model = ENNModel(
            enn,
            in_features,
            obs_dim,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
            predict_terminated=predict_terminated,
            dyn_dim=dyn_dim,
        )
        tx = optax.adamw(model_config["LR"], weight_decay=1e-4)
        not_prior_params = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))
        optimizer = nnx.Optimizer(model, tx, wrt=not_prior_params)
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            delta_next_state_loss=nnx.metrics.Average("delta_next_state_loss"),
            reward_loss=nnx.metrics.Average("reward_loss"),
            terminated_loss=nnx.metrics.Average("terminated_loss"),
        )
        return model, optimizer, metrics

    model, optimizer, metrics = build(keys)
    return model, (optimizer, metrics)


def _make_batched_train_model(update_steps, minibatch_size):
    def core(model, optimizer, metrics, dataset, pointer, rngs):
        return _train_model(
            model,
            optimizer,
            metrics,
            dataset,
            update_steps,
            pointer,
            minibatch_size,
            rngs,
        )

    batched = nnx.jit(nnx.vmap(core, in_axes=(0, 0, 0, 0, None, 0), out_axes=0))

    def train_fn(model, train_state, dataset, pointer, rngs):
        optimizer, metrics = train_state
        return batched(model, optimizer, metrics, dataset, pointer, rngs)

    return train_fn


register_model("enn")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_batched_train_model,
    }
)
