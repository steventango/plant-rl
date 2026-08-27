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
_CTL_TABLE_DAYS = 15
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

# --- Masked gate pinned to E21 Z11 (plant-data v29 e18e21 parquet, f=0.9) ------
# ``masked_log`` above gates against E18 Z11. E21's cohort outgrew E18's by ~56%
# by day 14 (day-14 mean area 5.25 vs 3.36), so the same 90% gate against E21 is
# a far harder target: 48.1% of the e18e21-daily-v29 transitions land off-target
# under it, versus 22.8% under E18. Nearly half the data therefore sits in the
# off-target penalty branch, which rewards *closing the gap* to the gate rather
# than saving energy — the two branches push intensity in opposite directions, so
# a harder gate shifts the policy toward more light.
#
# Regenerate with (never hand-edit; the baseline is not invariant to which
# experiments are in the join, because outlier detection is dataset-wide):
#   uv run --with polars python scripts/compute_masked_baseline.py \
#       --parquet /data/plant-rl/offline/v29/e18e21-daily-v29.parquet \
#       --experiment 21 --zone 11 --dataset plant-data/e18e21-daily-v29
_BASELINE_AREA_E21 = jnp.asarray(
    [
        0.2631085454201212,
        0.3124857643950437,
        0.41557147412536444,
        0.5546442148636799,
        0.6836382807517538,
        0.8475167410714285,
        1.0794577281517446,
        1.3925572308056287,
        1.7544857725805165,
        2.2485650510204085,
        2.6940398369169096,
        3.2178730867346936,
        3.821884110787172,
        4.460099160631994,
        5.24752162709842,
    ],
    dtype=jnp.float32,
)
_SURPLUS_LOG_MIN_E21, _SURPLUS_LOG_MAX_E21 = -4.02999452243816, -0.00014858325585986876

RewardMode = Literal[
    "area",
    "analytic",
    "analytic_diff",
    "masked_log",
    "masked_log_e21",
    "masked_growth",
    "learned",
]
_MASKED_MODES = frozenset({"masked_log", "masked_log_e21", "masked_growth"})
# Area-level gates and the Zone-11 baseline each is pinned to, with the
# off-target surplus range measured under that same baseline.
_LOG_GATES = {
    "masked_log": (_BASELINE_AREA, _SURPLUS_LOG_MIN, _SURPLUS_LOG_MAX),
    "masked_log_e21": (
        _BASELINE_AREA_E21,
        _SURPLUS_LOG_MIN_E21,
        _SURPLUS_LOG_MAX_E21,
    ),
}
# Differential setup: the dynamics model predicts the growth *advantage* over
# the same-batch, same-day control and the reward is that advantage minus an
# energy cost referenced to the control's own energy. Day-dependent, so its obs
# must carry the day.
_DIFF_MODES = frozenset({"analytic_diff"})
# Reward modes with no closed form: the reward comes from the dynamics model's
# reward head, fit to the dataset's own reward column. Required for the
# differential (in-batch control-normalized) reward of
# ``plant-data/e18e21-daily-diff-v29``:
#
#   R_t = (Δlog A_pol − Δlog A_ctl) − λ·(E_t − E_ctl)/E_ctl
#
# with the control resolved per (experiment, day) — E18 rows against E18 Z11,
# E21 rows against E21 Z11. The experiment id is not part of the observation, so
# no function of (obs, action, next_obs) can reproduce that reward; see the
# module docstring of :class:`PlantEnv` for the measured gap.
_LEARNED_MODES = frozenset({"learned"})


def reward_mode_requires_time(reward_mode: RewardMode) -> bool:
    """Whether ``reward_mode`` *needs* the day index in the observation.

    Masked modes gate on a day-dependent baseline and ``analytic_diff``
    references a per-day control, so their obs must carry time.
    ``learned`` needs it too: the differential reward it is used for subtracts a
    per-(experiment, day) control, so a reward head without the day can only fit
    the day-averaged control. Other modes may still opt in
    (``PlantEnv(include_time=True)``) to let the policy and dynamics model
    condition on the day.
    """
    return (
        reward_mode in _MASKED_MODES
        or reward_mode in _DIFF_MODES
        or reward_mode in _LEARNED_MODES
    )


def reward_mode_is_learned(reward_mode: RewardMode) -> bool:
    """Whether the reward comes from the model's reward head, not this env."""
    return reward_mode in _LEARNED_MODES


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

    For day-dependent reward modes the observation is ``[log_area, day]`` so the
    policy can condition on the day; otherwise it is ``[log_area]`` unless
    ``include_time`` opts in. The day comes from the dataset's ``wall_time``
    (see :mod:`offline_data`), not from the position in the episode.

    ``reward_mode="learned"`` has no oracle reward at all: ``compute_reward``
    raises and the reward is taken from the dynamics model's reward head, fit to
    the dataset's own reward column. That is the only faithful option for the
    differential reward of ``plant-data/e18e21-daily-diff-v29``. Measured on that
    dataset's 15411 transitions, reconstructing its stored reward from
    ``(growth, action, day)`` plus the true per-(experiment, day) control table
    still leaves a residual of std 0.106 against a reward whose own std is 0.081
    (R² < 0) — partly because the experiment id is unobserved, partly because
    plant-data drops gap-bridged rows so ``next_obs - obs`` across a kept pair is
    not always the one-day ``area_reward`` the stored reward was built from.
    """

    def __init__(
        self,
        act_dim: int,
        reward_mode: RewardMode = "analytic",
        include_time: bool = False,
        ctl_growth: jax.Array | None = None,
        ctl_energy: jax.Array | None = None,
        dyn_dim: int | None = None,
    ):
        self._act_dim = act_dim
        # Width of the dynamics output. Defaults to 1 (log-area only). It is a
        # property of the *trained checkpoint*, not of the env, so deployments
        # restoring a model trained before the day was dropped from the target
        # must pass the old width explicitly.
        self._dyn_dim = dyn_dim
        self._reward_mode: RewardMode = reward_mode
        self._include_time = include_time
        if reward_mode in _DIFF_MODES and (ctl_growth is None or ctl_energy is None):
            raise ValueError(
                f"reward_mode={reward_mode!r} needs the per-day control tables; "
                "pass ctl_growth/ctl_energy from load_offline_transitions."
            )
        self._ctl_growth = (
            None if ctl_growth is None else jnp.asarray(ctl_growth, dtype=jnp.float32)
        )
        self._ctl_energy = (
            None if ctl_energy is None else jnp.asarray(ctl_energy, dtype=jnp.float32)
        )

    @property
    def dyn_target(self) -> str:
        """What the dynamics model predicts for this reward mode.

        ``"advantage"`` for the differential setup: the target is
        ``Δlog A − Δlog A_ctl[experiment, day]``, which plant-data builds per row
        with the true experiment id. The shared unobserved batch factor is
        therefore already removed from the target, so the model does not need to
        observe the experiment to be unconfounded — and the action's effect is
        estimated free of the action/experiment correlation in the data.
        """
        return "advantage" if self._reward_mode in _DIFF_MODES else "delta_obs"

    @property
    def dyn_dim(self) -> int:
        """Number of dynamics outputs — the *stochastic* obs dims only.

        The day is part of the observation (day-dependent rewards need it) but is
        a deterministic clock that :meth:`compose_next_obs` sets, so it is never
        a prediction target: predicting it would waste capacity and put a
        deterministic quantity into the exploration bonus.

        Overridable for checkpoints trained before that change, which predicted
        the full observation delta.
        """
        return 1 if self._dyn_dim is None else self._dyn_dim

    def _next_day_index(self, day: jax.Array | int) -> jax.Array:
        """Control-table index for the day a transition *lands on*."""
        d = jnp.asarray(day, dtype=jnp.int32)
        return jnp.clip(d + 1, 0, _CTL_TABLE_DAYS - 1)

    def compose_next_obs(self, obs: jax.Array, dyn_pred: jax.Array) -> jax.Array:
        """Build the resulting observation from the model's dynamics output.

        For ``dyn_target == "advantage"`` the model predicts the growth advantage
        over the control, so the absolute area needs the control's own growth
        added back:

            log A_{t+1} = log A_t + ĝ_diff(s, a) + C(day + 1)

        ``C`` is indexed by *day* rather than by state on purpose. The day is
        exogenous — it advances no matter what the policy does — so ``C(day)`` is
        a fixed additive schedule the policy cannot steer. Indexing it by area
        would put a term the policy *can* influence, and which is not
        de-confounded, back into the controllable part of the dynamics. The
        plant's own size dependence is already carried by ``ĝ_diff``, which is
        fit at ``(log_area, day, action)``.

        Note ``C`` cancels out of the reward: ``compute_reward`` recomputes
        growth as ``next_obs - obs`` and subtracts ``C(day + 1)`` again, so the
        reward is exactly ``ĝ_diff - energy cost`` whatever ``C`` is. The choice
        of ``C`` therefore only shifts the simulated trajectory, never the reward
        at a given transition.
        """
        if self.dyn_dim >= self.obs_dim and self.obs_includes_time:
            # Legacy width: the model predicted the whole observation delta. Add
            # it, then overwrite the day, which is what the pre-advantage code did.
            obs_next = obs + dyn_pred
            day = jnp.asarray(obs[..., 1:2], dtype=obs_next.dtype) + 1.0
            return jnp.concatenate([obs_next[..., :1], day], axis=-1)
        area = obs[..., :1] + dyn_pred[..., :1]
        if self.dyn_target == "advantage":
            area = area + self._ctl_growth[self._next_day_index(obs[..., 1])][..., None]
        if not self.obs_includes_time:
            return area
        day = jnp.asarray(obs[..., 1:2], dtype=area.dtype) + 1.0
        return jnp.concatenate([area, day], axis=-1)

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
        mode = self._reward_mode
        if reward_mode_is_learned(mode):
            raise RuntimeError(
                f"reward_mode={mode!r} has no oracle reward — it is supplied by the "
                "model's reward head (main.py sets predict_reward_terminated and "
                "ModelEnvironment(oracle_reward=False) for it)."
            )
        growth = (next_obs - obs)[..., 0]
        # Current day: explicit ``time`` wins, else read it off the obs when it
        # carries one, else day 0 (reward-landscape plots pass neither).
        if time is not None:
            day = jnp.asarray(time, dtype=jnp.int32)
        elif self.obs_includes_time:
            day = jnp.asarray(obs[..., 1], dtype=jnp.int32)
        else:
            day = jnp.zeros((), dtype=jnp.int32)
        if mode == "area":
            return growth
        if mode == "analytic":
            # plant-data ``reward_linear``: area growth minus the *linear* energy
            # cost ``energy_reward_linear_1 = (energy - e_const) / e_const / N_STEPS``
            # (join_zones.py), with energy = power * hours (Wh) and N_STEPS = _EP_LEN.
            power = _P0 + _P_SLOPE * action[..., 0]
            energy = power * _HOURS
            return growth - (energy - _E_CONST_WH) / _E_CONST_WH / _EP_LEN

        if mode == "analytic_diff":
            # plant-data ``reward_diff``:
            #   (Δlog A − Δlog A_ctl) − (E − E_ctl)/E_ctl/N
            # Both terms reference the control on the *resulting* day, so the
            # action is scored against what the control did over the same
            # interval. Subtracting the control growth here undoes the add-back
            # in ``compose_next_obs``, leaving exactly the model's predicted
            # advantage — no dependence on the control table survives into the
            # reward.
            nd = self._next_day_index(day)
            ctl_energy = self._ctl_energy[nd]
            energy = (_P0 + _P_SLOPE * action[..., 0]) * _HOURS
            return (
                growth
                - self._ctl_growth[nd]
                - (energy - ctl_energy) / ctl_energy / _EP_LEN
            )

        day = jnp.clip(day, 0, _BASELINE_AREA.shape[0] - 1)
        # Both masked modes gate the resulting quantity against the next day's
        # baseline: masked_log gates the resulting area, masked_growth gates
        # the resulting growth.
        next_day = jnp.clip(day + 1, 0, _BASELINE_AREA.shape[0] - 1)
        energy = (_P0 + _P_SLOPE * action[..., 0]) * _HOURS
        energy_norm = (energy - _E_MIN_WH) / (_E_MAX_WH - _E_MIN_WH)
        on_reward = 1.0 - energy_norm

        if mode in _LOG_GATES:
            table, s_min, s_max = _LOG_GATES[mode]
            threshold = _MASK_F * table[next_day]
            clean_area = jnp.exp(next_obs[..., 0])
            on_target = clean_area >= threshold
            surplus = next_obs[..., 0] - jnp.log(threshold)
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
