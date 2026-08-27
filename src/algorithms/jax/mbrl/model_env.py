# Vendored from model-uncertainty-exploration (model_env.py); gymnax's generic
# Environment typing does not model our env/params subclasses, so structural
# type noise is silenced file-wide to keep the port byte-comparable upstream.
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
# pyright: reportIncompatibleMethodOverride=false, reportInvalidTypeArguments=false
# pyright: reportReturnType=false, reportArgumentType=false
# pyright: reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from algorithms.jax.mbrl.world_model import WorldModel

ResetSource = Literal["env", "buffer", "init"]


def reset_weights(
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
    count: jnp.ndarray | int,
    source: ResetSource,
) -> jnp.ndarray:
    """Sampling weights over replay-buffer rows for a dataset-driven reset.

    ``terminated``/``truncated`` have shape ``(..., N)`` (any leading batch dims).
    ``count`` is the number of valid (filled) buffer entries. Returns weights of
    the same shape, normalized along the last axis:

    - ``"buffer"``: uniform over all valid rows (MBPO branching from any state).
    - ``"init"``: uniform over episode-start rows — row 0 and any row whose
      predecessor was ``done`` — i.e. the dataset's initial-state distribution.
    """
    n = terminated.shape[-1]
    valid = jnp.arange(n) < count
    if source == "buffer":
        w = jnp.broadcast_to(valid, terminated.shape).astype(jnp.float32)
    elif source == "init":
        done = terminated | truncated
        first = jnp.ones_like(done[..., :1])
        prev_done = jnp.concatenate([first, done[..., :-1]], axis=-1)
        w = (prev_done & valid).astype(jnp.float32)
    else:
        raise ValueError(f"source must be 'buffer' or 'init', got {source!r}")
    return w / w.sum(axis=-1, keepdims=True)


@struct.dataclass
class ModelEnvState(environment.EnvState):
    obs: jnp.ndarray
    terminated: jnp.ndarray
    time: int
    z: jnp.ndarray
    last_action: jnp.ndarray


@struct.dataclass
class ModelEnvParams:
    env_params: environment.EnvParams
    max_steps_in_episode: int = struct.field(pytree_node=False)
    model: WorldModel | None = None
    alpha: float = 1.0
    beta: float = 0.0
    init_obs: jnp.ndarray | None = None  # per-seed replay buffer obs, (N, obs_dim)
    init_weights: jnp.ndarray | None = None  # per-seed reset sampling weights, (N,)

    def seed_vmap_axes(self) -> "ModelEnvParams":
        """vmap ``in_axes`` prefix that maps the per-seed ``model`` subtree and
        ``init_obs``/``init_weights`` buffers on axis 0; every other dynamic field
        is broadcast (None)."""
        return self.replace(
            env_params=None,
            model=0,
            alpha=None,
            beta=None,
            init_obs=0,
            init_weights=0,
        )

    def config_vmap_axes(self) -> "ModelEnvParams":
        """vmap ``in_axes`` prefix that maps the per-config ``alpha``/``beta``
        reward weights on axis 0; every other dynamic field is broadcast (None)."""
        return self.replace(
            env_params=None,
            model=None,
            alpha=0,
            beta=0,
            init_obs=None,
            init_weights=None,
        )


class ModelEnvironment(environment.Environment[ModelEnvState, ModelEnvParams]):
    def __init__(
        self,
        env: environment.Environment,
        env_params: environment.EnvParams,
        samples: int = 10,
        prediction_mode: Literal["mean", "sample"] = "mean",
        explore_bonus: Literal["std", "eig"] = "std",
        oracle_reward_terminated: bool = True,
        oracle_terminated: bool | None = None,
        uncertainty_dims: tuple[int, ...] | None = None,
        reset_source: ResetSource = "env",
        max_steps_in_episode: int | None = None,
        uncertainty_threshold: float | None = None,
    ):
        if prediction_mode not in ("mean", "sample"):
            raise ValueError(
                f"prediction_mode must be 'mean' or 'sample', got {prediction_mode!r}"
            )
        if explore_bonus not in ("std", "eig"):
            raise ValueError(
                f"explore_bonus must be 'std' or 'eig', got {explore_bonus!r}"
            )
        if reset_source not in ("env", "buffer", "init"):
            raise ValueError(
                f"reset_source must be 'env', 'buffer' or 'init', got {reset_source!r}"
            )
        self._real_env = env
        self._real_env_params = env_params
        self.samples = samples
        self.prediction_mode = prediction_mode
        self.explore_bonus = explore_bonus
        # ``oracle_reward_terminated`` sets both sources at once (the historical
        # coupled flag); ``oracle_terminated`` overrides just the termination
        # source. Splitting them is what lets an offline plant run take its
        # reward from the model's reward head (fit to the dataset's own reward
        # column) while keeping the oracle's deterministic day-count
        # termination: the plant datasets label 7.4% of steps terminal (bolting),
        # so a learned sigmoid termination head would cut rollouts short and
        # change the return scale on top of the reward change under test.
        self.oracle_reward = oracle_reward_terminated
        self.oracle_terminated = (
            oracle_reward_terminated if oracle_terminated is None else oracle_terminated
        )
        # Which model outputs the exploration bonus is allowed to look at. None
        # means every output, which is only right when every output is a
        # genuinely uncertain quantity. Restrict it to the stochastic dynamics
        # dims for plant: a deterministic day channel or a termination logit
        # contributes variance that has nothing to do with dynamics uncertainty,
        # and with alpha=0/beta=1 the explore policy's whole reward is this bonus.
        self._uncertainty_dims = (
            None if uncertainty_dims is None else jnp.asarray(uncertainty_dims)
        )
        self.reset_source = reset_source
        self._max_steps_override = max_steps_in_episode
        self.uncertainty_threshold = uncertainty_threshold
        self._zero_action = jnp.zeros_like(
            env.action_space(env_params).sample(jax.random.key(0))
        )

    @property
    def default_params(self) -> ModelEnvParams:
        return ModelEnvParams(
            env_params=self._real_env_params,
            max_steps_in_episode=(
                self._max_steps_override
                if self._max_steps_override is not None
                else self._real_env_params.max_steps_in_episode
            ),
        )

    def step(
        self,
        key: jax.Array,
        state: ModelEnvState,
        action: int | float | jax.Array,
        params: ModelEnvParams | None = None,
    ) -> tuple[
        jax.Array, ModelEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]
    ]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, info = self.step_env(
            key_step, state, action, params
        )
        truncated = state_st.time >= params.max_steps_in_episode
        if self.uncertainty_threshold is not None:
            # Dynamic uncertainty-based truncation: end the rollout early once
            # the model's per-step epistemic uncertainty exceeds the threshold.
            truncated = truncated | (info["uncertainty"] > self.uncertainty_threshold)
        done = terminated | truncated
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        info = {**info, "next_obs": obs_st}

        return obs, state, reward, terminated, truncated, info

    def reset(
        self, key: jax.Array, params: ModelEnvParams | None = None
    ) -> tuple[jax.Array, ModelEnvState]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params

        # Reset
        obs, state = self.reset_env(key, params)

        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: ModelEnvState,
        action: int | float | jax.Array,
        params: ModelEnvParams,
    ) -> tuple[jax.Array, ModelEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        model = params.model
        x = model.single_input(state.obs, action)
        if self.prediction_mode == "mean":
            y = model.predict_mean(x)
        else:
            y = model.predict_sample(x, state.z[0])
        # NOTE: the "eig" and "std" bonuses are on different scales and are NOT
        # magnitude-matched. "std" is the posterior standard deviation (linear,
        # in normalized output units) while "eig" is ½ log(1 + σ²_ep) nats
        # (logarithmic in variance). Consequently --beta is not directly
        # comparable across the two bonus types and must be retuned when
        # switching between them.
        r_intrinsic = model.uncertainty(
            x, state.z, self.explore_bonus, dims=self._uncertainty_dims
        )

        dyn_pred = model.denormalize_delta_obs(y[..., : model.dyn_dim])
        env_obs_space = self._real_env.observation_space(params.env_params)
        if hasattr(self._real_env, "compose_next_obs"):
            # The env owns the mapping from the model's dynamics output to the
            # next observation. It is not always "obs + delta": the plant
            # differential setup predicts a growth *advantage* and the env adds
            # the control's own growth back, and it sets the deterministic day
            # itself rather than predicting it.
            obs = self._real_env.compose_next_obs(state.obs, dyn_pred)
        else:
            obs = state.obs + dyn_pred
        obs = jnp.clip(obs, env_obs_space.low, env_obs_space.high)
        # Day-dependent envs carry their clock in the obs (see above), so an
        # oracle call must be reconstructed at that day, not at the step counter.
        oracle_time = (
            state.obs[..., 1]
            if getattr(self._real_env, "obs_includes_time", False)
            else state.time
        )
        if self.oracle_reward:
            if hasattr(self._real_env, "obs_to_reward_terminated"):
                r_oracle, term_oracle = self._real_env.obs_to_reward_terminated(
                    state.obs, action, obs
                )
            else:
                reconstructed = self._real_env.get_state(
                    state.obs, state.last_action, oracle_time, next_obs=obs
                )
                _, _, r_oracle, term_oracle, _ = self._real_env.step_env(
                    key, reconstructed, action, params.env_params
                )
            r_exploit = r_oracle
        elif self.oracle_terminated:
            # Termination only. Going through step_env would also evaluate the
            # oracle reward, which for a learned-reward mode does not exist, so
            # ask the terminal predicate about the *resulting* state directly —
            # is_terminal(time >= horizon) on time+1 matches step_env's
            # time + 1 >= horizon.
            if hasattr(self._real_env, "obs_to_reward_terminated"):
                _, term_oracle = self._real_env.obs_to_reward_terminated(
                    state.obs, action, obs
                )
            else:
                next_state = self._real_env.get_state(
                    obs, action, oracle_time + 1, next_obs=obs
                )
                term_oracle = self._real_env.is_terminal(
                    next_state, params.env_params
                )
            r_exploit = model.denormalize_reward(y[..., model.reward_index])
        else:
            r_exploit = model.denormalize_reward(y[..., model.reward_index])
        terminated = (
            term_oracle
            if self.oracle_terminated
            else jax.nn.sigmoid(y[..., model.terminated_index]) > 0.5
        )
        r = params.alpha * r_exploit + params.beta * r_intrinsic
        state = ModelEnvState(
            obs=obs,
            terminated=terminated,
            time=state.time + 1,
            z=state.z,
            last_action=jnp.asarray(action),
        )
        return obs, state, r, terminated, {"uncertainty": r_intrinsic}

    def reset_env(
        self, key: jax.Array, params: ModelEnvParams
    ) -> tuple[jax.Array, ModelEnvState]:
        model = params.model
        key, key_reset, key_z, key_idx = jax.random.split(key, 4)
        if self.reset_source == "env":
            obs, _ = self._real_env.reset_env(key_reset, params.env_params)
        else:
            # "buffer" or "init": sample a buffer row by the supplied weights.
            idx = jax.random.choice(
                key_idx, params.init_obs.shape[0], p=params.init_weights
            )
            obs = params.init_obs[idx]
        z = model.sample_index(key_z, self.samples)
        state = ModelEnvState(
            obs=obs,
            terminated=jnp.bool_(False),
            time=0,
            z=z,
            last_action=self._zero_action,
        )
        return obs, state

    def get_obs(self, state: ModelEnvState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        return state.obs

    def is_terminal(self, state: ModelEnvState, params: ModelEnvParams) -> jax.Array:
        """Check whether state transition is terminal."""
        return state.terminated

    def discount(self, state: ModelEnvState, params: ModelEnvParams) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._real_env.num_actions

    def action_space(self, params: ModelEnvParams):
        """Action space of the environment."""
        return self._real_env.action_space(params.env_params)

    def observation_space(self, params: ModelEnvParams):
        """Observation space of the environment."""
        return self._real_env.observation_space(params.env_params)

    def state_space(self, params: ModelEnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
                "obs": self._real_env.observation_space(params.env_params),
                "terminated": spaces.Box(
                    low=False, high=True, shape=(), dtype=jnp.bool
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
