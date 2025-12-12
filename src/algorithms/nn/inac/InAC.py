import dataclasses
from pathlib import Path
from typing import Any, Dict, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import minari
import numpy as np
import optax
from flax import nnx
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from algorithms.nn.inac.agent.base import load
from algorithms.nn.inac.agent.in_sample import (
    ActorCritic,
    Hypers,
    Optimizers,
    _policy,
    _update_beta,
    _update_pi,
    _update_q,
    _update_value,
    polyak_update,
)


class InAC(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        # Enable planning by default if not explicitly set
        # This allows InAC to work with the offline training interface
        if "use_planning" not in params:
            params = {**params, "use_planning": True}

        super().__init__(observations, actions, params, collector, seed)

        # ------------------------------
        # -- Configuration Parameters --
        # ------------------------------
        self.discrete_control = params.get("discrete_control", False)
        self.policy_type = params.get("policy_type", "dirichlet")
        self.hidden_units = params.get("hidden_units", 256)
        self.learning_rate = params.get("learning_rate", 1e-4)
        self.actor_lr_scale = params.get("actor_lr_scale", 1)
        self.tau = params.get("tau", 0.001)
        self.polyak = params.get("polyak", 0.995)
        self.target_network_update_freq = params.get("target_network_update_freq", 1)
        self.use_target_network = bool(params.get("use_target_network", 1))
        self.weight_decay = params.get("weight_decay", 1e-4)
        self.clip_grad_norm = params.get("clip_grad_norm", None)
        self.deterministic = params.get("deterministic", False)
        self.batch_size = params.get("batch_size", 256)
        self.update_freq = params.get("update_freq", 1)
        self.updates_per_step = params.get("updates_per_step", 0)

        # Path to pre-trained model (optional)
        self.pretrained_path = params.get("pretrained_path", None)

        # Path to offline dataset to load into buffer (optional)
        self.offline_dataset_name = params.get("offline_dataset", None)

        # Ensure observations is a flat dimension
        if isinstance(observations, tuple):
            if len(observations) == 1:
                self.state_dim = observations[0]
            else:
                self.state_dim = int(np.prod(observations))
        else:
            self.state_dim = observations

        self.action_dim = actions

        # -------------------------
        # -- Initialize Networks --
        # -------------------------
        self.rngs = nnx.Rngs(seed)
        self.actor_critic = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=self.hidden_units,
            discrete_control=self.discrete_control,
            policy_type=self.policy_type,
            rngs=self.rngs,
        )

        # ---------------
        # -- Optimizer --
        # ---------------
        critic_adamw = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        actor_adamw = optax.adamw(
            self.actor_lr_scale * self.learning_rate, weight_decay=self.weight_decay
        )

        if self.clip_grad_norm is not None:
            critic_adamw = optax.chain(
                optax.clip_by_global_norm(self.clip_grad_norm), critic_adamw
            )
            actor_adamw = optax.chain(
                optax.clip_by_global_norm(self.clip_grad_norm), actor_adamw
            )

        self.optimizers: Optimizers = Optimizers(
            pi=nnx.Optimizer(
                self.actor_critic.pi,
                actor_adamw,
                wrt=nnx.Param,
            ),
            q=nnx.Optimizer(
                self.actor_critic.q,
                critic_adamw,
                wrt=nnx.Param,
            ),
            value=nnx.Optimizer(
                self.actor_critic.value_net,
                critic_adamw,
                wrt=nnx.Param,
            ),
            beh_pi=nnx.Optimizer(
                self.actor_critic.beh_pi,
                actor_adamw,
                wrt=nnx.Param,
            ),
        )

        # -------------------
        # -- Hyperparameters --
        # -------------------
        self.hypers = Hypers(
            batch_size=self.batch_size,
            eps=1e-8,
            tau=self.tau,
            gamma=self.gamma,
            polyak=self.polyak,
            exp_threshold=10000,
            use_target_network=self.use_target_network,
            target_network_update_freq=self.target_network_update_freq,
        )

        # ------------------
        # -- Load Pretrained Model --
        # ------------------
        if self.pretrained_path:
            self._load_pretrained(self.pretrained_path)

        # --------------------------
        # -- Stateful information --
        # --------------------------
        self.steps = 0
        self.updates = 0

        # Store current state and action for online learning
        self.current_state = None
        self.current_action = None

        # Replay buffer for online learning using Flashbax
        self.max_buffer_size = params.get("max_buffer_size", 10000)

        dummy_state = np.zeros(self.state_dim, dtype=np.float32)
        dummy_action = (
            np.zeros(self.action_dim, dtype=np.float32)
            if not self.discrete_control
            else np.int32(0)
        )
        dummy_transition = {
            "state": dummy_state,
            "action": dummy_action,
            "reward": np.float32(0.0),
            "next_state": dummy_state,
            "termination": np.float32(0.0),
        }

        self.replay_buffer = fbx.make_flat_buffer(
            max_length=self.max_buffer_size,
            min_length=self.batch_size,
            sample_batch_size=self.batch_size,
        )
        self.replay_buffer = dataclasses.replace(
            self.replay_buffer,
            init=jax.jit(self.replay_buffer.init),
            add=jax.jit(self.replay_buffer.add, donate_argnums=0),
            sample=jax.jit(self.replay_buffer.sample),
            can_sample=jax.jit(self.replay_buffer.can_sample),
        )
        self.replay_state = self.replay_buffer.init(dummy_transition)

        # Normalization parameters
        self.state_min = None
        self.state_diff = None

        # Load offline dataset into buffer if specified
        if self.offline_dataset_name:
            offline_dataset = minari.load_dataset(self.offline_dataset_name)
            self.load_normalization_params(offline_dataset.observation_space)
            self.load(offline_dataset)

    def _load_pretrained(self, path: str):
        """Load pre-trained InAC model from disk."""
        parameters_dir = Path(path)
        if not parameters_dir.exists():
            raise FileNotFoundError(f"Pretrained model path not found: {path}")

        # Get the graph definitions (structure without parameters)
        actor_critic_graphdef, _ = nnx.split(self.actor_critic)
        optimizers_graphdef, _ = nnx.split(self.optimizers)

        # Load the saved state and merge back with structure
        loaded_ac, loaded_opt = load(
            actor_critic_graphdef, optimizers_graphdef, parameters_dir
        )

        # Replace the current objects with loaded ones
        self.actor_critic: ActorCritic = loaded_ac  # type: ignore
        self.optimizers: Optimizers = loaded_opt  # type: ignore

        print(f"Successfully loaded pre-trained model from {path}")

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Min-max normalize observations."""
        if self.state_min is None:
            return obs
        return (obs - self.state_min) / self.state_diff

    def load_normalization_params(self, observation_space):
        low = observation_space.low
        high = observation_space.high

        # Start with identity normalization
        self.state_min = np.zeros_like(low)
        self.state_diff = np.ones_like(low)

        # Check for finite bounds
        mask = ~np.logical_or(np.isinf(low), np.isinf(high))

        # For finite dimensions, set min and diff
        self.state_min[mask] = low[mask]
        self.state_diff[mask] = high[mask] - low[mask]

        # Avoid division by zero if high == low
        self.state_diff[self.state_diff == 0] = 1.0

    def load(self, dataset: minari.MinariDataset):
        """
        Load offline dataset into the replay buffer.

        This allows the agent to continue learning from the offline data
        while also adding new online experiences.

        Args:
            dataset: MinariDataset to load into the buffer
        """

        # Collect all transitions
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_terminations = []

        for episode in dataset.iterate_episodes():
            episode_length = len(episode.observations) - 1
            for t in range(episode_length):
                all_states.append(self._normalize(episode.observations[t]))
                all_actions.append(episode.actions[t])
                all_rewards.append(episode.rewards[t])
                all_next_states.append(self._normalize(episode.observations[t + 1]))
                all_terminations.append(episode.terminations[t])

        # Convert to JAX arrays
        dataset_transitions = {
            "state": jnp.array(all_states, dtype=jnp.float32),
            "action": jnp.array(
                all_actions,
                dtype=jnp.float32 if not self.discrete_control else jnp.int32,
            ),
            "reward": jnp.array(all_rewards, dtype=jnp.float32),
            "next_state": jnp.array(all_next_states, dtype=jnp.float32),
            "termination": jnp.array(all_terminations, dtype=jnp.float32),
        }

        # Add all transitions to buffer using scan for efficiency
        def add_transition(replay_state, transition):
            replay_state = self.replay_buffer.add(replay_state, transition)
            return replay_state, None

        self.replay_state, _ = jax.lax.scan(
            add_transition, self.replay_state, dataset_transitions
        )

        print(f"Loaded {len(all_states)} transitions from offline dataset")

    def policy(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute action probabilities or sample action.

        For discrete actions: returns action probabilities
        For continuous actions: returns sampled action
        """
        obs = np.asarray(obs)
        obs = self._normalize(obs)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)

        obs_jax = jnp.array(obs)
        action = _policy(self.actor_critic.pi, obs_jax, self.deterministic, self.rngs)
        action = jax.device_get(action)

        if len(action.shape) > 1:
            action = action[0]

        return action

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, x: np.ndarray, extra: Dict[str, Any] | None = None):  # type: ignore
        """Start an episode."""
        if extra is None:
            extra = {}

        x = np.asarray(x)
        a = self.policy(x)

        # Store for online learning
        self.current_state = self._normalize(x)
        self.current_action = a

        # For discrete actions, return integer action
        if self.discrete_control:
            a = int(a)

        return a, {}

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]):  # type: ignore
        """Take a step in the environment."""
        a = -1

        # Store transition for online learning
        if (
            xp is not None
            and self.current_state is not None
            and self.current_action is not None
        ):
            terminal = False
            self._store_transition(
                self.current_state, self.current_action, r, xp, terminal
            )

        # Sample next action
        if xp is not None:
            xp = np.asarray(xp)
            a = self.policy(xp)

            # Store for next transition
            self.current_state = self._normalize(xp)
            self.current_action = a

            # For discrete actions, return integer action
            if self.discrete_control:
                a = int(a)

        # Update if enabled
        info = {}
        if self.updates_per_step > 0:
            info = self.update()

        return a, info

    def end(self, r: float, extra: Dict[str, Any]):  # type: ignore
        """End an episode."""
        # Store terminal transition
        if self.current_state is not None and self.current_action is not None:
            terminal = True
            xp = np.zeros(self.observations)
            self._store_transition(
                self.current_state, self.current_action, r, xp, terminal
            )

        # Update if enabled
        info = {}
        if self.updates_per_step > 0:
            info = self.update()

        # Reset episode state
        self.current_state = None
        self.current_action = None

        return info

    def _store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ):
        """Store a transition in the Flashbax buffer."""
        # Ensure proper types for JAX
        if isinstance(action, int):
            action_jax = jnp.int32(action)
        else:
            action_jax = jnp.array(action, dtype=jnp.float32)

        transition = {
            "state": jnp.array(state, dtype=jnp.float32),
            "action": action_jax,
            "reward": jnp.float32(reward),
            "next_state": jnp.array(next_state, dtype=jnp.float32),
            "termination": jnp.float32(terminal),
        }

        # Add to buffer
        self.replay_state = self.replay_buffer.add(self.replay_state, transition)

    def update(self):
        """Perform online updates if enough data is available."""
        self.steps += 1

        # Only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return {}

        # Skip updates if buffer isn't large enough
        if not self.replay_buffer.can_sample(self.replay_state):
            return {}

        # Perform multiple updates per step if configured
        info = {}
        for _ in range(self.updates_per_step):
            info = self._update()
        return info

    def _update(self):
        """Perform a single update step using data from the Flashbax buffer."""
        # Sample a batch from the Flashbax buffer using JIT-compiled function
        batch = self.replay_buffer.sample(self.replay_state, self.rngs.replay_sample())

        # Perform update steps
        loss_beta = _update_beta(
            self.actor_critic.beh_pi, self.optimizers.beh_pi, batch
        )
        loss_vs, v_info, logp_info = _update_value(
            self.actor_critic.value_net,
            self.optimizers.value,
            self.actor_critic.pi,
            self.actor_critic.q_target,
            self.hypers.tau,
            batch,
            self.rngs,
        )
        loss_q, qinfo = _update_q(
            self.actor_critic.q,
            self.optimizers.q,
            self.actor_critic.pi,
            self.actor_critic.q_target,
            self.hypers.gamma,
            self.hypers.tau,
            batch,
            self.rngs,
        )
        loss_pi = _update_pi(
            self.actor_critic.pi,
            self.optimizers.pi,
            self.actor_critic.q,
            self.actor_critic.value_net,
            self.actor_critic.beh_pi,
            self.hypers.eps,
            self.hypers.exp_threshold,
            self.hypers.tau,
            batch,
        )

        # Sync target networks if needed
        if (
            self.use_target_network
            and self.updates % self.target_network_update_freq == 0
        ):
            self.actor_critic.pi_target = polyak_update(
                self.actor_critic.pi, self.actor_critic.pi_target, 1 - self.polyak
            )
            self.actor_critic.q_target = polyak_update(
                self.actor_critic.q, self.actor_critic.q_target, 1 - self.polyak
            )

        self.updates += 1

        # Collect metrics if collector is available
        if self.collector:
            self.collector.collect("beta_loss", float(jax.device_get(loss_beta)))
            self.collector.collect("actor_loss", float(jax.device_get(loss_pi)))
            self.collector.collect("critic_loss", float(jax.device_get(loss_q)))
            self.collector.collect("value_loss", float(jax.device_get(loss_vs)))

        return {
            "beta": float(jax.device_get(loss_beta)),
            "actor": float(jax.device_get(loss_pi)),
            "critic": float(jax.device_get(loss_q)),
            "value": float(jax.device_get(loss_vs)),
        }

    def plan(self):  # type: ignore
        """
        Perform a single training update step (for offline training).
        This method fits the RL-Glue interface for planning/updating.

        Returns:
            Dictionary with loss values: {"beta", "actor", "critic", "value"}
        """
        if not self.replay_buffer.can_sample(self.replay_state):
            return {}
        return self._update()

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        """Get state for checkpointing."""
        # Get the base state
        base_state = super().__getstate__()

        # Add InAC-specific state
        actor_critic_state = nnx.split(self.actor_critic)[1]
        optimizers_state = nnx.split(self.optimizers)[1]

        base_state.update(
            {
                "actor_critic_state": actor_critic_state,
                "optimizers_state": optimizers_state,
                "steps": self.steps,
                "updates": self.updates,
                "replay_state": self.replay_state,
                "current_state": self.current_state,
                "current_action": self.current_action,
            }
        )

        return base_state

    def __setstate__(self, state):
        """Restore state from checkpoint."""
        # Restore base state
        super().__setstate__(state)

        # Restore InAC-specific state
        actor_critic_graphdef, _ = nnx.split(self.actor_critic)
        optimizers_graphdef, _ = nnx.split(self.optimizers)

        self.actor_critic = nnx.merge(
            actor_critic_graphdef, state["actor_critic_state"]
        )
        self.optimizers = nnx.merge(optimizers_graphdef, state["optimizers_state"])

        self.steps = state["steps"]
        self.updates = state["updates"]
        self.replay_state = state.get("replay_state", self.replay_state)
        self.current_state = state.get("current_state", None)
        self.current_action = state.get("current_action", None)
