# Experiment E22 / Phase P1 — Differential vs standard reward (Z1–Z5)

## Status: staged locally, NOT deployed

Configs, checkpoints and tests are prepared on aurora only. Nothing has been
copied to the deploy host and `compose.yml` is untouched — archcraft runs live
experiments and those steps are done manually.

Zone numbers for the agent arms (Z1–Z5) are a **proposal** mirroring the E21/P1
layout; confirm them against the actual E22 chamber allocation before use.
`Constant11.json` is byte-identical to `E22/P0/Constant11.json`, so Z11 simply
continues its P0 policy into P1.

Follows **E22/P0** (constant-100 incubation, transplant 2026-08-24, agent start
2026-08-30) on branch `deploy-E22`, which also carries the alliance-zone05
lightbar swap + recalibration, CV at 1-hour frequency, and `iqm_log_clean_area`
logging.

## Overview

E22/P1 tests whether the **differential** reward produces a different (and
better) policy than the **standard** absolute reward. Both are trained offline in
`model-uncertainty-exploration` on `plant-data` v29 datasets.

- **standard** — reward = log-area growth − linear energy cost.
  `plant-data/e18e21-daily-v29`, `runs/20260827/plant_plant-data_e18e21-daily-v29/gamma0.8_enn_ln_diffsetup_analytic`
- **differential** — reward = growth *advantage over the same-batch, same-day
  constant-white control*, minus an energy cost referenced to that control's own
  energy:
  `R_t = (Δlog A − Δlog A_ctl) − λ·(E_t − E_ctl)/E_ctl`.
  `plant-data/e18e21-daily-diff-v29`, `runs/20260827/plant_plant-data_e18e21-daily-diff-v29/gamma0.8_enn_ln_diffsetup_analytic_diff`
- **masked_e21** — day-gated reward: the *resulting* area is gated against **90%
  of E21 Z11's** per-day mean area; on-target pays `1 − energy_norm` (so: save
  energy), off-target pays a normalized surplus in `[-1, 0]` (so: close the gap).
  `plant-data/e18e21-daily-v29`, `reward_mode: masked_log_e21`,
  `runs/20260827/plant_plant-data_e18e21-daily-v29/gamma0.8_enn_ln_diffsetup_masked_log_e21`
- **all_data** — standard reward over all four experiments (E18–E21), as a
  data-quantity reference.
  `plant-data/all-daily-v29`, `runs/20260827/plant_plant-data_all-daily-v29/gamma0.8_enn_ln_diffsetup_analytic`

Z1, Z2 and Z3 all come from the **same parquet** (`e18e21-daily-v29`) and differ
only in the reward, so differences between them are attributable to the reward
definition. Z4 is the single exploration arm; Z3 became a third *exploit* reward
variant rather than a second explore arm, because both explore heads are
bang-bang regardless of reward (see caveat 1) and two of them would be
indistinguishable.

## Configs

| Zone | Config | Agent | Checkpoint | Head | Selection | obs_dim | Nightly retrain |
|---|---|---|---|---|---|---|---|
| Z1 | `Z1.json` | `PPOPolicy1` | standard | `ppo_eval` (exploit) | mean | 2 | — |
| Z2 | `Z2.json` | `PPOPolicy2` | differential | `ppo_eval` (exploit) | mean | 2 | — |
| Z3 | `Z3.json` | `PPOPolicy3` | masked_e21 | `ppo_eval` (exploit) | mean | 2 | — |
| Z4 | `Z4.json` | `AdaptivePPOPolicy4` | differential | `ppo_explore` | sampled | 2 | `analytic_diff` |
| Z5 | `Z5.json` | `PPOPolicy5` | all_data | `ppo_eval` (exploit) | mean | 2 | — |
| Z11 | `Constant11.json` | `ConstantAgent` | — (constant white) | — | — | — | — |

**Z11 is the in-batch control** — constant balanced white at
`constant_action: 1.0` (100 PPFD), the same fixed policy E18 Z11 and E21 Z11 ran.
That match is what makes the differential reward's reference meaningful: the v29
control tables were built from those zones at measured intensity ≈0.9955, so E22
Z11 at 1.0 is directly comparable. It keeps `enable_cv_pipeline: false` per the
constant-zone convention, which does **not** prevent the control's growth from
being recovered: flash photography still captures the daily standardized image and
plant-data's vision pipeline extracts `clean_area` offline. E21 Z11 ran with the
same flag and still supplied v29's E21 control column.

The five agent zones: timezone `Etc/GMT-2`, `observation: log_area`, `action: intensity`,
`action_timestep: 720`, actions clipped to `[0.4, 1.3]`, flash photography and
night enforcement on, CV pipeline enabled.

**All three arms use `obs_dim: 2`** (`[log_area, day_index]`). The absolute arms
were trained with `--include-time` on purpose: otherwise the standard-vs-
differential contrast would be confounded with a different state space. This
differs from E21/P1, where the analytic arm was `obs_dim: 1`.

## How the differential setup works

The dynamics model regresses `area_reward_diff` — the growth advantage — instead
of the observation delta. plant-data builds that column **per row using the true
experiment id**, which is what makes it de-confounded: the batch factor is shared
by everything in a run, so subtracting the same run's control removes it. The
model therefore never needs to observe the experiment.

`PlantEnv.compose_next_obs` adds the per-day control growth `C(day+1)` back to
recover an absolute area, and `compute_reward` subtracts it again, so **`C`
cancels out of the reward exactly** — verified in the m-u-e test suite by
shifting `C` and confirming the reward is unchanged. `C` is indexed by *day*, not
state, because the day is exogenous: the policy cannot steer it, so `C(day)` is a
fixed additive schedule rather than a term the policy can game.

Reconstruction against the dataset's own `reward_diff` column: **R² 0.9975**
(residual std 0.004, which is just the analytic `E(I)` energy fit).

The dynamics output is **1-wide** (log-area only). The day is an input but not a
target: it is deterministic, so predicting it wasted capacity and — because the
explore arm's reward *is* the epistemic bonus — contributed roughly a quarter of
that bonus as uncertainty about a counter. No arm predicts termination either;
termination is just `day >= 14`, supplied by the oracle.

## The E21-pinned masked gate

The stock `masked_log` gate is pinned to **E18** Z11. E21's cohort outgrew E18's
by roughly 56% by day 14 (day-14 mean area 5.25 vs 3.36), so pinning to E21 at the
same `f = 0.9` is a much harder target:

| resulting day | gate, E18-pinned | gate, E21-pinned |
|---|---|---|
| 1 | −1.219 | −1.269 |
| 7 | −0.040 | **+0.226** |
| 14 | +1.108 | **+1.552** |

At day 14 it demands `e^0.44` ≈ **1.56×** more area. Off-target fraction over the
training transitions goes from **22.8%** (E18-pinned) to **48.1%** (E21-pinned),
so nearly half the data sits in the penalty branch. That matters because the two
branches push intensity in *opposite* directions — on-target rewards saving
energy, off-target rewards closing the gap — so a harder gate shifts the policy
toward more light while below the gate.

The trained policy behaves exactly that way. Over a (log-area × day) grid its
mean action pushes light when below the gate and drops to the 0.4 floor once
above it:

| day | gate | below the gate | above |
|---|---|---|---|
| 0 | −1.44 | 0.85–1.02 | 0.40 |
| 7 | +0.23 | rising to 1.24 just under | 0.40 |
| 14 | +1.55 | 1.30 | 0.40 |

Constants were regenerated from the same parquet the policy trains on, via
`scripts/compute_masked_baseline.py` in m-u-e. Do not reuse constants across
dataset versions: `transform_outlier_detection` runs dataset-wide, so adding an
experiment to the join shifts the surviving rows and the per-day means. The E18
constants still in `plant_env.py` came from the v28 `mixed-e18-e19-e20` parquet
and differ from v29's by up to 0.048 (~1.6%).

## Caveats you should read before deploying

**1. The explore policy is bang-bang, not exploration.** Driving the real config
path over a simulated 14-day window (`test_via_main_real.py`) gives, for daytime
PPFD:

| Zone | unique daytime PPFD levels | range |
|---|---|---|
| Z1 standard exploit | 9 | 40 – 118 |
| Z2 differential exploit | 12 | **76 – 111** |
| Z3 masked_e21 exploit | 1 — only `40` on this trajectory | 40 |
| Z4 differential **explore** | **2** — only `40` and `130` | 40 – 130 |
| Z5 all_data exploit | 7 | 40 – 130 |

Z4 emits nothing but the two clip bounds. Z3's single level is *not* degeneracy —
the replayed mock trajectory sits above the E21 gate the whole time, so the policy
correctly coasts at minimum light; on the grid above it is fully graded. But the
same thing will happen on hardware if E22's plants track or exceed E21 Z11, in
which case Z3 becomes a low-light zone. **Check E22/P0's current areas against the
E21 Z11 curve before starting**, and lower `f` if E22 is running smaller or
larger than expected. E22 data has not synced to aurora yet (only E20/E21 are
there), so I could not check this locally.

The exploit heads are graded, and
Z2 notably never touches either rail — it holds a moderate band, which is the
cleanest single piece of evidence that the differential policy is better behaved
than the standard one. The explore heads' mean action also falls from ~0.88 at day
0 to ~0.45 by day 7, and the `all_data` explore head is nearly deterministic
(σ ≈ 0.010). Deployed as-is, Z3 and Z4 are two zones alternating between minimum
and maximum light. The cause is not yet understood; the leading
hypothesis is that the bonus is cumulative over the rollout, so suppressing growth
while the day counter advances walks the state into "small plant, late day", which
is unvisited because area and day form a narrow diagonal band in the data. Removing
the day and termination heads from the bonus did **not** fix it. If the point of
Z3/Z4 is exploration rather than a low-light control, this should be resolved
first.

**2. E22 has no in-batch control in the agent's view.** The differential reward's
premise is a control running in the *same* batch. Online, `AdaptivePPOPolicy` has
no access to another zone's data, so `analytic_diff` retraining falls back to the
per-day control table shipped in the offline buffer — i.e. **E18/E21's** control.
E22's own batch effect is therefore not removed by the nightly retrain. Because
the subtraction is constant in the action it cannot distort the action ranking
within a day, only the weighting across days, so this degrades gracefully rather
than breaking. Two consequences:

- Z11 is included for exactly this reason: E22's own control **exists in the data**
  even though the online agent cannot read it, which makes E22 de-confoundable
  offline afterwards — what actually matters for the analysis. Only one control
  zone is defined, matching how v29 pinned a single reference zone per experiment
  (E21 Z12 was constant white too but deliberately excluded).
- Making the online retrain properly in-batch needs the agent to read its own
  run's constant-white zone live. That is unbuilt.

**3. Offline and online rows use slightly different controls.** Offline rows were
de-confounded with each row's own `(experiment, day)` control; online rows use the
pooled per-day table. The two differ by up to ~0.066 in log-area growth — about
0.8σ of the reward — so the retrain target level is mildly inconsistent between
the two halves of the buffer.

**4. `all_data` cannot serve the differential arm.** 51% of `all-daily-v29` rows
are E19/E20, which have no constant-white zone, so `area_reward_diff` is NaN and
the loader drops them — leaving essentially just E18+E21. Z5 is therefore
standard-reward only, and the "more data" comparison exists only for the absolute
reward.

**5. The absolute arm saturates the floor.** Its day-14 action is exactly 0.400
for all 10 training seeds, so that end of its schedule is clipped rather than
measured.

## Staging and tests

```
./experiments/online/E22/P1/ship_checkpoints.sh      # stages checkpoints/E22/P1
JAX_PLATFORMS=cpu uv run pytest tests/algorithms/test_E22P1_parity.py
```

`test_via_main_real.py` drives all five configs through the real loader +
`AsyncRLGlue` against the mock chamber and renders one action-over-time PDF per
zone into `figures/`; nightly retrain is pushed out of range there and covered by
the parity suite instead.

`test_E22P1_parity.py` asserts config conventions, that the staged buffers match
their policies (obs_dim, data-derived action bounds, 1-wide target, per-experiment
control subtraction, and that only the adaptive arm ships a buffer), golden action
parity for all five zones plus Z11's control config, and — the important one for a
new reward mode — that the explore arm completes a nightly retrain and actually
**swaps** (`_retrain_count == 1`). `plan()` swallows exceptions and still
returns to phase `idle`, so reaching `idle` is not evidence of success.

Goldens are dumped by m-u-e `scripts/dump_golden_actions.py --experiment E22`.

The vendored `src/algorithms/jax/mbrl/` port was re-synced from m-u-e for this
change (`world_model.py`, `enn.py`, `model_env.py`, `plant_env.py`). E20/P1 and
E21/P1 remain restorable: `dyn_dim` now comes from the staged buffer, and buffers
exported before it existed fall back to the old full-observation-delta width.
