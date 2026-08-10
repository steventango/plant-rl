# Experiment E21 / Phase P1 — Offline-trained PPO deployment (Z1–Z4)

## Overview

Agent-controlled treatment phase following the P0 constant-100 incubation
(transplant 2026-08-04, agent start 2026-08-10). Four zones deploy PPO policies
trained offline in `model-uncertainty-exploration` on the minari dataset
`plant-data/visu-v28` (N=12143 transitions), in two reward variants × two heads:

- **analytic** — reward = log-area growth − linear energy cost
  (`runs/20260807/plant_plant-data_visu-v28/gamma0.8_enn_ln`, obs = `[log_area]`)
- **masked_log** — day-gated reward: the *resulting* area (next obs) is gated
  against the *next* day's Zone-11 baseline, with the off-target penalty on the
  log-area surplus
  (`runs/20260731/plant_plant-data_visu-v28/gamma0.8_enn_ln_masked_log`,
  obs = `[log_area, day_index]`)

Each checkpoint carries two heads: `ppo_eval` (**exploit** — pure extrinsic
reward, alpha=1/beta=0) and `ppo_explore` (**explore** — pure ENN
information-gain bonus, alpha=0/beta=1).

## Configs

| Zone | Config | Agent | Checkpoint | Head | Action selection | obs_dim | Nightly retrain |
|---|---|---|---|---|---|---|---|
| Z1 (zone01) | `Z1.json` | `PPOPolicy1` | analytic | `ppo_eval` (exploit) | mean | 1 | — |
| Z2 (zone02) | `Z2.json` | `PPOPolicy2` | masked_log | `ppo_eval` (exploit) | mean | 2 | — |
| Z3 (zone03) | `Z3.json` | `AdaptivePPOPolicy3` | analytic | `ppo_explore` | sampled | 1 | `reward_mode: analytic` |
| Z4 (zone04) | `Z4.json` | `AdaptivePPOPolicy4` | masked_log | `ppo_explore` | sampled | 2 | `reward_mode: masked_log` |

All four: timezone `Etc/GMT-2`, `observation: log_area`, `action: intensity`,
`action_timestep: 720` (one decision per 12 h photoperiod), actions clipped to
`[0.4, 1.3]` (E19/E20 ops convention; the v28 action space is `[0.381, 1.3]`),
flash photography + night enforcement on, and the **CV pipeline enabled** —
unlike the E21/P0 constant zones, which disable it deliberately.

The masked_log zones observe `[log_area, day_index]`, where the day index is the
count of 09:00 polls since the episode began, clamped to 14 (the training
episode length). The `log_area` observation is the interquartile mean of
per-plant `log(clean_area)`, which drops dead plants and CV failures.

Z3/Z4 (`AdaptivePPOPolicy`) additionally update their ENN world model and
retrain a fresh explore policy **every night after 21:00 local, on CPU**, using
the v28 offline dataset plus the per-plant transitions collected since (one per
pot per daily poll, matched by `pot_id`, reward-IQR filtered). The retrain runs
in bounded slices inside the RlGlue `plan()` hook so it can never delay the
09:00 poll, and aborts (keeping the current policy) if unfinished by 08:00.

## Retrain archives (Z3/Z4)

After each completed nightly retrain the adaptive arms archive the artifacts to
`retrain_archive_dir` (under the bind-mounted `/data/plant-rl`, so they persist
across restarts and are readable from the host):

```
/data/plant-rl/online/E21/P1/<Agent>/<zone>/retrain_archive/
    <YYYY-MM-DD>_retrain0001/
        network/     orbax checkpoint of the swapped-in policy
        obs_norm/    its observation normalizer
        model/       the updated ENN world model
        online_transitions.npz   the per-plant data collected since deploy
        metadata.json            counts, reward_mode, ppo/model steps
```

The checkpoints use the same layout `main.py` writes, so the training repo's
tooling loads them directly — e.g. to replay a night's policy or diff successive
world models. Archiving is best-effort: a failure is logged and never costs the
already-completed retrain. Omit `retrain_archive_dir` to disable.

## Checkpoints

Not in git (`checkpoints/` is gitignored). Stage with
`./experiments/online/E21/P1/ship_checkpoints.sh`:

```
checkpoints/E21/P1/analytic/checkpoint/{ppo_explore,ppo_eval,model}/...
checkpoints/E21/P1/analytic/offline_transitions.npz
checkpoints/E21/P1/masked_log/checkpoint/{ppo_explore,ppo_eval,model}/...
checkpoints/E21/P1/masked_log/offline_transitions.npz
```

## Verification

- `uv run pytest tests/algorithms/test_E21P1_parity.py` — every zone agent
  reproduces the golden actions dumped from its checkpoint
  (`tests/test_data/E21P1/golden_actions.json`, regenerate with
  `model-uncertainty-exploration/scripts/dump_golden_actions.py --experiment E21`),
  and both adaptive arms complete a real nightly retrain (asserted via
  `_retrain_count`, since `plan()` swallows exceptions and returns to idle
  either way).
- `uv run python experiments/online/E21/P1/test_via_main_real.py` — mock-chamber
  dry-run of all four configs (no hardware, wandb disabled).
- After deployment: confirm each zone's first 09:00 action against the golden,
  and that Z3/Z4 log `Nightly retrain complete ... (completed retrains: N)`
  with a nonzero `online` transition count after the first full day.

## Deployment

Point compose services `zone1`–`zone4` at these configs and
`docker compose up -d zone1 zone2 zone3 zone4`.
