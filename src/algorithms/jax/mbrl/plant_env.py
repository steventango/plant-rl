# Vendored from model-uncertainty-exploration (plant_env.py); gymnax's generic
# Environment typing does not model our env/params subclasses, so structural
# type noise is silenced file-wide to keep the port byte-comparable upstream.
# Plotting-only landscape helpers are stripped; everything else is verbatim.
# pyright: reportIncompatibleMethodOverride=false, reportInvalidTypeArguments=false
# pyright: reportReturnType=false
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


_P0, _P_SLOPE, _E_CONST_WH, _HOURS = 3.3, 44.9, 571.6, 12.0
_EP_LEN = 14.0

# Masked-reward constants from plant-data join_zones on the mixed-e18-e19-e20
# subsampled-daily-v28 parquet (f=0.9). Off-target uses min-max of surplus over
# *off-target rows only*, shifted to [-1, 0]. The Zone-11 baselines below are
# pinned to E18 Z11 and so are unchanged from the v27 parquet; only the energy
# range and the surplus ranges move with the added E20 data.
_MASK_F = 0.9
_E_MIN_WH = 260.484666666669
_E_MAX_WH = 798.6427500000027
_BASELINE_AREA = jnp.asarray(
    [
        0.2845042592930029,
        0.3284751958819242,
        0.39459030479991536,
        0.4607054137179065,
        0.5444416299786043,
        0.6836635044642857,
        0.8751163645901909,
        1.067246196613726,
        1.3329055916721526,
        1.584624743852459,
        1.9470720207725947,
        2.3683627075317832,
        2.693766793346437,
        3.0285379335706057,
        3.3639775088462804,
    ],
    dtype=jnp.float32,
)
# Day 0 has no growth baseline in Zone 11; fill with day-1 so t=0 is defined.
_BASELINE_GROWTH = jnp.asarray(
    [
        0.1421117135405597,
        0.1421117135405597,
        0.15580959582492587,
        0.16950747810929206,
        0.1605959316147196,
        0.22425637565680395,
        0.23983826627754173,
        0.19295724752200866,
        0.21528826847931462,
        0.15564769239622256,
        0.2212170144759219,
        0.18150758488490198,
        0.1418429854491252,
        0.10674979721340086,
        0.08709036905965564,
    ],
    dtype=jnp.float32,
)
# Off-target-only surplus ranges (max ≈ 0 just below the gate), from
# scripts/recompute_masked_surplus.py over the visu-v28 training transitions.
# Both modes gate against the *next* day's baseline: masked_log gates the
# resulting area (next_obs vs the next-day area baseline, off-target next-obs
# rows only — excludes each trajectory's day-0 start, which is never a
# resulting area); masked_growth gates the resulting growth against the
# next day's growth baseline.
_SURPLUS_LOG_MIN, _SURPLUS_LOG_MAX = -3.548095226287842, -2.7120113372802734e-06
_SURPLUS_GROWTH_MIN, _SURPLUS_GROWTH_MAX = -1.5482633113861084, -3.904104232788086e-06

RewardMode = Literal["area", "analytic", "masked_log", "masked_growth"]
_MASKED_MODES = frozenset({"masked_log", "masked_growth"})


def reward_mode_requires_time(reward_mode: RewardMode) -> bool:
    """Whether ``reward_mode`` *needs* the day index in the observation.

    Masked modes gate on a day-dependent baseline, so their obs must carry time.
    Other modes may still opt in (``PlantEnv(include_time=True)``) to let the
    policy and dynamics model condition on the day.
    """
    return reward_mode in _MASKED_MODES


@struct.dataclass
class PlantEnvState(environment.EnvState):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    time: int


@struct.dataclass
class PlantEnvParams:
    area_min: float
    area_max: float
    act_low: jnp.ndarray
    act_high: jnp.ndarray
    max_steps_in_episode: int


class PlantEnv(environment.Environment[PlantEnvState, PlantEnvParams]):
    """Offline plant-growth environment.

    This env does not simulate its own dynamics: ``step_env`` replays the
    ``next_obs`` already stored on the state rather than computing a transition.
    It is therefore only meaningful when driven by a ``ModelEnvironment`` that
    supplies predicted transitions and uses the oracle reward (the env's
    ``compute_reward``); it has no standalone ``default_params``.

    For masked reward modes the observation is ``[log_area, time]`` so the
    policy can condition on the day-dependent gate; otherwise it is
    ``[log_area]`` unless ``include_time`` opts in.
    """

    def __init__(
        self,
        act_dim: int,
        reward_mode: RewardMode = "analytic",
        include_time: bool = False,
    ):
        self._act_dim = act_dim
        self._reward_mode: RewardMode = reward_mode
        self._include_time = include_time

    @property
    def obs_includes_time(self) -> bool:
        """True when obs is ``[log_area, time]``.

        Forced on for masked modes, which cannot compute their gate without the
        day; otherwise controlled by ``include_time``.
        """
        return self._include_time or reward_mode_requires_time(self._reward_mode)

    @property
    def obs_dim(self) -> int:
        return 2 if self.obs_includes_time else 1

    @property
    def default_params(self) -> PlantEnvParams:
        raise RuntimeError(
            "PlantEnv has no default params — construct PlantEnvParams "
            "from the loaded dataset and pass it explicitly."
        )

    def observation_space(self, params: PlantEnvParams) -> spaces.Box:
        if self.obs_includes_time:
            return spaces.Box(
                low=jnp.asarray([params.area_min, 0.0], dtype=jnp.float32),
                high=jnp.stack(
                    [
                        jnp.asarray(params.area_max, dtype=jnp.float32),
                        jnp.asarray(params.max_steps_in_episode, dtype=jnp.float32),
                    ]
                ),
                shape=(2,),
                dtype=jnp.float32,
            )
        return spaces.Box(
            low=params.area_min,
            high=params.area_max,
            shape=(1,),
            dtype=jnp.float32,
        )

    def action_space(self, params: PlantEnvParams) -> spaces.Box:
        return spaces.Box(
            low=params.act_low,
            high=params.act_high,
            shape=(self._act_dim,),
            dtype=jnp.float32,
        )

    def _pack_obs(self, area: jax.Array, time: jax.Array | int) -> jax.Array:
        area = jnp.asarray(area, dtype=jnp.float32).reshape(-1)[:1]
        if not self.obs_includes_time:
            return area
        t = jnp.asarray(time, dtype=jnp.float32).reshape(())
        return jnp.concatenate([area, t[None]])

    def reset_env(
        self, key: jax.Array, params: PlantEnvParams
    ) -> tuple[jax.Array, PlantEnvState]:
        """Reset to a log-area sampled uniformly within the observed range.

        The dataset-driven initial-state distribution now lives in
        :func:`model_env.reset_weights` (``reset_source="init"``); offline runs
        use that instead, so this is only a generic in-bounds fallback.
        """
        area = jax.random.uniform(
            key, (1,), minval=params.area_min, maxval=params.area_max
        )
        obs = self._pack_obs(area, 0)
        state = PlantEnvState(obs=obs, next_obs=jnp.zeros_like(obs), time=0)
        return obs, state

    def get_state(
        self,
        obs: jax.Array,
        last_action: jax.Array | None = None,
        time: int | None = None,
        next_obs: jax.Array | None = None,
    ) -> PlantEnvState:
        assert time is not None
        assert next_obs is not None
        return PlantEnvState(obs=obs, next_obs=next_obs, time=time)

    def step_env(
        self,
        key: jax.Array,
        state: PlantEnvState,
        action: jax.Array,
        params: PlantEnvParams,
    ) -> tuple[jax.Array, PlantEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        next_obs = state.next_obs
        reward = self.compute_reward(state.obs, action, next_obs, time=state.time)
        terminated = jnp.asarray(state.time + 1 >= params.max_steps_in_episode)
        new_state = PlantEnvState(
            obs=next_obs, next_obs=jnp.zeros_like(next_obs), time=state.time + 1
        )
        return next_obs, new_state, reward, terminated, {}

    def compute_reward(
        self,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
        time: jax.Array | int | None = None,
    ) -> jax.Array:
        """Oracle reward for model-based rollouts and visualization.

        ``time`` is the current day index (0..14). Masked modes evaluate the
        *resulting* area (``next_obs``) against the *next* day's Zone-11 baseline
        (``time + 1``), so the action is credited for the area it produces; when
        ``time`` is omitted (e.g. reward-landscape plots) day 0 is used.
        """
        growth = (next_obs - obs)[..., 0]
        mode = self._reward_mode
        if mode == "area":
            return growth
        if mode == "analytic":
            # plant-data ``reward_linear``: area growth minus the *linear* energy
            # cost ``energy_reward_linear_1 = (energy - e_const) / e_const / N_STEPS``
            # (join_zones.py), with energy = power * hours (Wh) and N_STEPS = _EP_LEN.
            power = _P0 + _P_SLOPE * action[..., 0]
            energy = power * _HOURS
            return growth - (energy - _E_CONST_WH) / _E_CONST_WH / _EP_LEN

        day = jnp.asarray(0 if time is None else time, dtype=jnp.int32)
        day = jnp.clip(day, 0, _BASELINE_AREA.shape[0] - 1)
        # Both masked modes gate the resulting quantity against the next day's
        # baseline: masked_log gates the resulting area, masked_growth gates
        # the resulting growth.
        next_day = jnp.clip(day + 1, 0, _BASELINE_AREA.shape[0] - 1)
        energy = (_P0 + _P_SLOPE * action[..., 0]) * _HOURS
        energy_norm = (energy - _E_MIN_WH) / (_E_MAX_WH - _E_MIN_WH)
        on_reward = 1.0 - energy_norm

        if mode == "masked_log":
            baseline = _BASELINE_AREA[next_day]
            threshold = _MASK_F * baseline
            clean_area = jnp.exp(next_obs[..., 0])
            on_target = clean_area >= threshold
            surplus = next_obs[..., 0] - jnp.log(threshold)
            s_min, s_max = _SURPLUS_LOG_MIN, _SURPLUS_LOG_MAX
        elif mode == "masked_growth":
            baseline = _BASELINE_GROWTH[next_day]
            threshold = _MASK_F * baseline
            on_target = growth >= threshold
            surplus = growth - threshold
            s_min, s_max = _SURPLUS_GROWTH_MIN, _SURPLUS_GROWTH_MAX
        else:
            raise ValueError(f"Unknown reward_mode: {mode!r}")

        off_reward = (surplus - s_min) / (s_max - s_min) - 1.0
        return jnp.where(on_target, on_reward, off_reward)

    def get_obs(self, state: PlantEnvState, params=None, key=None) -> jax.Array:
        return state.obs

    def is_terminal(self, state: PlantEnvState, params: PlantEnvParams) -> jax.Array:
        return state.time >= params.max_steps_in_episode

    def state_space(self, params: PlantEnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "obs": self.observation_space(params),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def num_actions(self) -> int:
        return self._act_dim
