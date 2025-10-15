import os
import time

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from minari import MinariDataset


def fill_offline_data_to_buffer(dataset: MinariDataset, batch_size: int):
    dataset_size = dataset.total_steps
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_terminations = []

    for episode in dataset.iterate_episodes():
        all_obs.append(episode.observations[:-1])
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)
        all_next_obs.append(episode.observations[1:])
        all_terminations.append(episode.terminations)

    dataset_transitions = {
        "state": jnp.concatenate(all_obs, axis=0),
        "action": jnp.concatenate(all_actions, axis=0),
        "reward": jnp.concatenate(all_rewards, axis=0),
        "next_state": jnp.concatenate(all_next_obs, axis=0),
        "termination": jnp.concatenate(all_terminations, axis=0),
    }

    dummy_transition = jax.tree_util.tree_map(lambda x: x[0], dataset_transitions)
    replay = fbx.make_flat_buffer(
        max_length=dataset_size,
        min_length=batch_size,
        sample_batch_size=batch_size,
    )
    replay_state = replay.init(dummy_transition)
    add_fn = jax.jit(replay.add, donate_argnums=(0,))

    def add_transition(carry, transition):
        replay_state = carry
        replay_state = add_fn(replay_state, transition)
        return replay_state, None

    replay_state, _ = jax.lax.scan(add_transition, replay_state, dataset_transitions)
    return replay, replay_state


def populate_returns(eval_env, _policy, pi, timeout: int, total_ep, rngs: nnx.Rngs):
    ep_returns = np.zeros(total_ep)
    for episode in range(total_ep):
        ep_return = eval_episode(eval_env, _policy, pi, timeout, rngs=rngs)
        ep_returns[episode] = ep_return
    return ep_returns


def eval_episode(eval_env, _policy, pi, timeout: int, rngs):
    state = eval_env.reset()
    total_rewards = 0
    ep_steps = 0
    done = False
    while True:
        action = eval_step(_policy, pi, state, rngs=rngs)
        state, reward, done, _ = eval_env.step([action])
        total_rewards += reward
        ep_steps += 1
        if done or ep_steps == timeout:
            break

    return total_rewards


def log_return(logger, total_steps, returns, name, elapsed_time):
    total_episodes = len(returns)
    mean, median, min_, max_ = (
        np.mean(returns),
        np.median(returns),
        np.min(returns),
        np.max(returns),
    )

    logger.info(
        f"{name} LOG: steps {total_steps}, episodes {total_episodes:3d}, "
        f"returns {mean:.2f}/{median:.2f}/{min_:.2f}/{max_:.2f}/{len(returns)} (mean/median/min/max/num), {elapsed_time:.2f} steps/s"
    )
    return mean, median, min_, max_


def evaluate(logger, total_steps, eval_env, _policy, pi, timeout, total_ep, rngs):
    t0 = time.time()
    returns = populate_returns(eval_env, _policy, pi, timeout, total_ep, rngs)
    elapsed_time = time.time() - t0
    log_return(
        logger,
        total_steps,
        returns,
        "TEST",
        elapsed_time,
    )
    normalized = np.array(
        [eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in returns]
    )
    mean, median, min_, max_ = log_return(
        logger, total_steps, normalized, "Normalized", elapsed_time
    )
    return mean, median, min_, max_


def eval_step(_policy, pi, state: np.ndarray, rngs: nnx.Rngs):
    state = jnp.asarray(state)
    a = _policy(pi, state, deterministic=True, rngs=rngs)
    return np.asarray(a)


def save(module: nnx.Module, optimizers: nnx.Module, parameters_dir):
    module_state = nnx.split(module)[1]
    optimizers_state = nnx.split(optimizers)[1]

    ckpt = {
        "module": module_state,
        "optimizers": optimizers_state,
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(os.path.join(parameters_dir, "default"), ckpt, force=True)


def load(module: nnx.Module, optimizers: nnx.Module, parameters_dir):
    with ocp.StandardCheckpointer() as checkpointer:
        ckpt = checkpointer.restore(os.path.join(parameters_dir, "default"))
    module = nnx.merge(module, ckpt["module"])
    optimizers = nnx.merge(optimizers, ckpt["optimizers"])
    return module, optimizers
