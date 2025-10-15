import time
from functools import partial
from pathlib import Path

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from algorithms.nn.inac.agent.base import evaluate, fill_offline_data_to_buffer, save
from algorithms.nn.inac.network.network_architectures import (
    DoubleCriticDiscrete,
    DoubleCriticNetwork,
    FCNetwork,
)
from algorithms.nn.inac.network.policy_factory import MLPCont, MLPDiscrete


@nnx.jit
def polyak_update(new, old, step_size):
    new_params = nnx.state(new)
    old_params = nnx.state(old)
    polyak_params = optax.incremental_update(new_params, old_params, step_size)
    nnx.update(old, polyak_params)
    return old


class ActorCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: int,
        discrete_control: bool,
        rngs: nnx.Rngs,
    ) -> None:
        if discrete_control:
            self.pi = MLPDiscrete(state_dim, action_dim, [hidden_units] * 2, rngs=rngs)
            self.beh_pi = MLPDiscrete(
                state_dim, action_dim, [hidden_units] * 2, rngs=rngs
            )
            self.q = DoubleCriticDiscrete(
                state_dim, [hidden_units] * 2, action_dim, rngs=rngs
            )
        else:
            self.pi = MLPCont(state_dim, action_dim, [hidden_units] * 2, rngs=rngs)
            self.beh_pi = MLPCont(state_dim, action_dim, [hidden_units] * 2, rngs=rngs)
            self.q = DoubleCriticNetwork(
                state_dim, action_dim, [hidden_units] * 2, rngs=rngs
            )
        self.value_net = FCNetwork(
            jnp.prod(state_dim), [hidden_units] * 2, 1, rngs=rngs
        )
        self.pi_target = nnx.clone(self.pi)
        self.q_target = nnx.clone(self.q)


@chex.dataclass
class Optimizers:
    beh_pi: nnx.Optimizer
    pi: nnx.Optimizer
    q: nnx.Optimizer
    value: nnx.Optimizer


@chex.dataclass
class Hypers:
    batch_size: float
    eps: float
    exp_threshold: float
    gamma: float
    polyak: float
    target_network_update_freq: int
    tau: float
    use_target_network: bool


@nnx.jit
def _update_beta(beh_pi, beh_pi_optimizer: nnx.Optimizer, batch):
    def loss_fn(pi):
        log_probs = pi.get_logprob(
            batch.experience.first["state"], batch.experience.first["action"]
        )
        return -log_probs.mean()

    loss, grads = nnx.value_and_grad(loss_fn)(beh_pi)
    beh_pi_optimizer.update(beh_pi, grads)
    return loss


@nnx.jit
def _update_value(
    value_net, value_optimizer: nnx.Optimizer, pi, q_target, tau, batch, rngs: nnx.Rngs
):
    def loss_fn(value_net, rngs):
        v_phi = value_net(batch.experience.first["state"]).squeeze(-1)
        actions, log_probs = pi(batch.experience.first["state"], rngs=rngs)
        min_Q, _, _ = q_target(batch.experience.first["state"], actions)
        target = min_Q - tau * log_probs
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        return value_loss, (v_phi, log_probs)

    (loss, (v_phi, log_probs)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        value_net, rngs
    )
    value_optimizer.update(value_net, grads)
    return loss, v_phi, log_probs


@nnx.jit
def _update_q(
    q_net: nnx.Module,
    q_optimizer: nnx.Optimizer,
    pi: nnx.Module,
    q_target: nnx.Module,
    gamma: float,
    tau: float,
    batch,
    rngs: nnx.Rngs,
):
    def loss_fn(q_net, rngs):
        next_actions, log_probs = pi(batch.experience.second["state"], rngs=rngs)
        min_Q, _, _ = q_target(batch.experience.second["state"], next_actions)
        q_target_values = batch.experience.first["reward"] + gamma * (
            1 - batch.experience.first["termination"]
        ) * (min_Q - tau * log_probs)

        min_q, q1, q2 = q_net(
            batch.experience.first["state"], batch.experience.first["action"]
        )

        critic1_loss = (0.5 * (q_target_values - q1) ** 2).mean()
        critic2_loss = (0.5 * (q_target_values - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        return loss_q, min_q

    (loss, q_info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_net, rngs)
    q_optimizer.update(q_net, grads)
    return loss, q_info


@nnx.jit
def _update_pi(
    pi,
    pi_optimizer: nnx.Optimizer,
    q,
    value_net,
    beh_pi,
    eps: float,
    exp_threshold: float,
    tau: float,
    batch,
):
    def loss_fn(pi):
        log_probs = pi.get_logprob(
            batch.experience.first["state"], batch.experience.first["action"]
        )
        min_Q, _, _ = q(
            batch.experience.first["state"], batch.experience.first["action"]
        )
        value = value_net(batch.experience.first["state"]).squeeze(-1)
        beh_log_prob = beh_pi.get_logprob(
            batch.experience.first["state"], batch.experience.first["action"]
        )

        clipped = jnp.clip(
            jnp.exp((min_Q - value) / tau - beh_log_prob),
            eps,
            exp_threshold,
        )
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss

    loss, grads = nnx.value_and_grad(loss_fn)(pi)
    pi_optimizer.update(pi, grads)
    return loss


@partial(nnx.jit, static_argnums=(2,))
def _policy(pi, o, deterministic, rngs):
    a, _ = pi(o, deterministic=deterministic, rngs=rngs)
    return a


@nnx.jit
def _sync_target(
    pi,
    pi_target,
    q,
    q_target,
    polyak,
    target_network_update_freq,
    use_target_network,
    total_steps,
):
    def sync(pi, pi_target, q, q_target):
        pi_target = polyak_update(pi, pi_target, 1 - polyak)
        q_target = polyak_update(q, q_target, 1 - polyak)
        return pi_target, q_target

    def no_sync(pi, pi_target, q, q_target):
        return pi_target, q_target

    pi_target, q_target = nnx.cond(
        jnp.logical_and(
            use_target_network,
            total_steps % target_network_update_freq == 0,
        ),
        sync,
        no_sync,
        pi,
        pi_target,
        q,
        q_target,
    )
    return pi_target, q_target


@nnx.jit
def _update(
    ac: ActorCritic,
    optimizers: Optimizers,
    hypers: Hypers,
    batch,
    step,
    rngs,
):
    loss_beta = _update_beta(ac.beh_pi, optimizers.beh_pi, batch)
    loss_vs, v_info, logp_info = _update_value(
        ac.value_net, optimizers.value, ac.pi, ac.q_target, hypers.tau, batch, rngs
    )
    loss_q, qinfo = _update_q(
        ac.q, optimizers.q, ac.pi, ac.q_target, hypers.gamma, hypers.tau, batch, rngs
    )
    loss_pi = _update_pi(
        ac.pi,
        optimizers.pi,
        ac.q,
        ac.value_net,
        ac.beh_pi,
        hypers.eps,
        hypers.exp_threshold,
        hypers.tau,
        batch,
    )
    _sync_target(
        ac.pi,
        ac.pi_target,
        ac.q,
        ac.q_target,
        hypers.polyak,
        hypers.target_network_update_freq,
        hypers.use_target_network,
        step,
    )
    return {
        "beta": loss_beta,
        "actor": loss_pi,
        "critic": loss_q,
        "value": loss_vs,
        "q_info": qinfo.mean(),
        "v_info": v_info.mean(),
        "logp_info": logp_info.mean(),
    }


CarryType = tuple[
    ActorCritic,
    Optimizers,
    fbx.trajectory_buffer.BufferState,
    Hypers,
    nnx.Rngs,
]


def get_train_func(max_steps: int, replay: fbx.trajectory_buffer.TrajectoryBuffer):
    @nnx.scan(length=max_steps, in_axes=(nnx.Carry, 0))
    def train_scan(carry: CarryType, step):
        actor_critic, optimizers, replay_state, hypers, rngs = carry
        batch = replay.sample(replay_state, rngs.replay_sample())
        losses = _update(actor_critic, optimizers, hypers, batch, step, rngs)
        return (actor_critic, optimizers, replay_state, hypers, rngs), losses

    return train_scan


def train(
    discrete_control: bool,
    state_dim: int,
    action_dim: int,
    hidden_units: int,
    learning_rate: float,
    tau: float,
    polyak: float,
    exp_path: Path,
    seed: int,
    gamma: float,
    offline_data,
    batch_size,
    use_target_network,
    target_network_update_freq,
    logger,
    max_steps,
    log_interval,
    weight_decay: float,
):
    rngs = nnx.Rngs(seed)
    actor_critic = ActorCritic(
        state_dim,
        action_dim,
        hidden_units,
        discrete_control,
        rngs=rngs,
    )
    optimizers = Optimizers(
        pi=nnx.Optimizer(
            actor_critic.pi,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        q=nnx.Optimizer(
            actor_critic.q,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        value=nnx.Optimizer(
            actor_critic.value_net,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        beh_pi=nnx.Optimizer(
            actor_critic.beh_pi,
            optax.adamw(learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
    )
    hypers = Hypers(
        batch_size=batch_size,
        eps=1e-8,
        tau=tau,
        gamma=gamma,
        polyak=polyak,
        exp_threshold=10000,
        use_target_network=use_target_network,
        target_network_update_freq=target_network_update_freq,
    )

    replay, replay_state = fill_offline_data_to_buffer(
        offline_data, batch_size=hypers.batch_size
    )

    train_func = get_train_func(log_interval, replay)

    carry = (actor_critic, optimizers, replay_state, hypers, rngs)
    eval_env = env_fn()

    total_steps = 0
    i = 0
    evaluations = np.zeros(max_steps // log_interval + 1)
    while True:
        if log_interval and not total_steps % log_interval:
            actor_critic = carry[0]
            mean, *_ = evaluate(
                logger,
                total_steps,
                eval_env,
                _policy,
                actor_critic.pi,
                timeout,
                5,
                rngs,
            )
            evaluations[i] = mean
            i += 1

        if total_steps >= max_steps:
            break

        steps = jnp.arange(total_steps, total_steps + log_interval)

        t0 = time.time()
        carry, losses = train_func(carry, steps)
        jax.block_until_ready(losses)
        elapsed_time = log_interval / (time.time() - t0)
        total_steps += log_interval
        logger.info(
            f"TRAIN LOG: steps {total_steps}, {total_steps * 100 // max_steps}%, {elapsed_time:.2f} steps/s"
        )

    np.save(exp_path / "evaluations.npy", np.array(evaluations))
    actor_critic = carry[0]
    save(actor_critic, optimizers, exp_path / "parameters")
