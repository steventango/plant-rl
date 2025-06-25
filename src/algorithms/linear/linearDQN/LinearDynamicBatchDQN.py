from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch

from algorithms.linear.linearDQN.linear_NNAgent import LinearNNAgent
from algorithms.linear.linearDQN.linear_networks import LinearNetworkBuilder
from utils.jax import huber_loss

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu


@cxu.dataclass
class AgentState:
    params: Any
    target_params: Any
    optim: optax.OptState


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


class LinearDynamicBatchDQN(LinearNNAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = params["target_refresh"]
        self.min_batch_size = params["min_batch"]

        self.state = AgentState(
            params=self.state.params,
            target_params=self.state.params,
            optim=self.state.optim,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: LinearNetworkBuilder) -> None:
        self.q = builder.addHead(lambda: hk.Linear(self.actions, name="q"))

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):
        # phi = self.phi(state.params, x).out
        # return self.q(state.params, phi)
        return self.q(state.params, x)

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        self.plan()

    def plan(self):
        super().plan()
        # skip updates if the min batch size hasn't been reached yet
        if self.buffer.size() <= self.min_batch_size:
            return

        for _ in range(self.updates_per_step):
            # Sample min(batch_size, buffer_size) transitions so that we can do still do updates
            # before collecting batch_size samples when batch_size is large
            batch = self.buffer.sample(min(self.buffer.size(), self.batch_size))
            weights = self.buffer.isr_weights(batch.eid)
            self.state, metrics = self._computeUpdate(self.state, batch, weights)

            metrics = jax.device_get(metrics)

            priorities = metrics["delta"]
            self.buffer.update_batch(batch, priorities=priorities)

            for k, v in metrics.items():
                self.collector.collect(k, np.mean(v).item())

            self.updates += 1

            if self.updates % self.target_refresh == 0:
                self.state.target_params = self.state.params

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            optim=optim,
        )

        return new_state, metrics

    def _loss(
        self, params: hk.Params, target: hk.Params, batch: Batch, weights: jax.Array
    ):
        # phi = self.phi(params, batch.x).out
        # phi_p = self.phi(target, batch.xp).out

        qs = self.q(params, batch.x)
        qsp = self.q(target, batch.xp)

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, batch.a, batch.r, batch.gamma, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        return loss, metrics
