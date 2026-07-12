# Experiment E20 / Phase P1 — Offline-trained PPO deployment (Z1–Z5, Z11)

## Overview

Treatment phase following the P0 constant-100 incubation. Six zones deploy PPO
policies trained offline in `model-uncertainty-exploration` (branch
`feat/classic-plan-every`) on the minari dataset `plant-data/visu-v2-v27`, with
two reward variants:

- **analytic** — reward = log-area growth − linear energy cost
  (checkpoint `runs/20260710/plant_plant-data_visu-v2-v27/gamma0.8_enn_ln`,
  policy obs = `[log_area]`)
- **masked** — day-gated reward vs the Zone-11 baseline
  (checkpoint `runs/20260711/plant_plant-data_visu-v2-v27/gamma0.8_enn_ln_masked_timeobs`,
  policy obs = `[log_area, day_index]`; the agent synthesizes the day index
  from 09:00 polls since start, clamped to 14)

## Configs

| Zone | Config | Agent | Checkpoint | Head | Action selection | Nightly retrain |
|---|---|---|---|---|---|---|
| Z1 (zone01) | `Z1.json` | `PPOPolicy1` | analytic | `ppo_eval` | mean | — |
| Z2 (zone02) | `Z2.json` | `PPOPolicy2` | masked | `ppo_eval` | mean | — |
| Z3 (zone03) | `Z3.json` | `PPOPolicy3` | analytic | `ppo_explore` | sample | — |
| Z4 (zone04) | `Z4.json` | `PPOPolicy4` | masked | `ppo_explore` | sample | — |
| Z5 (zone05) | `Z5.json` | `AdaptivePPOPolicy5` | analytic | `ppo_explore` | sample | `reward_mode: analytic` |
| Z11 (zone11) | `Z11.json` | `AdaptivePPOPolicy11` | masked | `ppo_explore` | sample | `reward_mode: masked` |
| Z12 (zone12) | `Z12.json` | `AdaptivePPOPolicy12` | masked | `ppo_explore` | sample | `reward_mode: masked` — **hardware derisk of Z11** (masked path: obs_dim 2, day-index, masked retrain), timezone `America/Edmonton`, CV pipeline re-enabled |

Explore heads were trained purely on the ENN's EIG exploration bonus
(alpha=0, beta=1); eval heads purely on the extrinsic reward. Sampled zones
draw one stochastic action per day with a key derived from `(seed, local
date)`, so a same-day container restart reproduces the same action. Actions
are clipped to `[0.4, 1.3]` (E19 ops convention; the dataset action space is
`[0.381, 1.3]`).

Z5/Z11 (`AdaptivePPOPolicy`, `src/algorithms/jax/AdaptivePPOPolicy.py`) update
their ENN world model and retrain a fresh explore policy inside the model
environment **every night after 21:00, on CPU**, using the offline dataset
(`offline_transitions.npz`) plus the transitions collected so far. The retrain
runs in bounded slices inside the RlGlue `plan()` hook so it can never delay a
09:00 poll, and aborts (keeping the old policy) if still unfinished at 08:00.

## Checkpoints

The frozen artifacts are NOT in git (`checkpoints/` is gitignored). Stage them
with `./ship_checkpoints.sh` (see the script for the exact rsync commands)
before starting the containers:

```
checkpoints/E20/P1/analytic/checkpoint/{ppo_explore,ppo_eval,model}/...
checkpoints/E20/P1/analytic/offline_transitions.npz
checkpoints/E20/P1/masked/checkpoint/{ppo_explore,ppo_eval,model}/...
checkpoints/E20/P1/masked/offline_transitions.npz
```

## Verification

- `uv run pytest tests/algorithms/test_PPOPolicy_parity.py` — asserts each
  agent reproduces the golden actions dumped from the training repo
  (`tests/test_data/E20P1/golden_actions.json`, regenerate with
  `model-uncertainty-exploration/scripts/dump_golden_actions.py`).
- `uv run python experiments/online/E20/P1/test_via_main_real.py` — mock-chamber
  dry-run of all six configs (no hardware, wandb disabled).
- After deployment, spot-check each zone's first 09:00 action against the
  golden values, and confirm Z5/Z11 log a completed retrain after 21:00.

## Z12 hardware derisk (run BEFORE the real deploy)

`Z12.json` is a copy of the Z5 treatment (analytic explore + nightly retrain)
bound to alliance-zone12 with `timezone: America/Edmonton` and the CV pipeline
re-enabled, used purely to prove on real hardware that the agent, the nightly
retrain, and the hourly agent checkpoint coexist safely before Z1–Z5/Z11 go
live. `tests/algorithms/test_PPOPolicy_parity.py::TestZ12Derisk` pins Z12's
agent-relevant params to Z5's so the rehearsal cannot drift from the real
thing.

Pre-flight (no hardware): `python experiments/online/E20/P1/derisk_z12_via_main_real.py`
forces the nightly cycle to overlap live mock stepping and asserts no
`glue.step` stalls, retrain completion + swap, and a mid-retrain agent pickle.
Pass `--full` for the production-hyperparameter rehearsal (~15–20 min).

On archcraft: run zone12 against `experiments/online/E20/P1/Z12.json`
(`uv run python src/main_real.py -e experiments/online/E20/P1/Z12.json -i 0 --deploy`
or point the zone12 compose command at it). Watch for: the 09:00 (Edmonton)
action logged in the zone's `raw_*.csv`, the `plant_rl.AdaptivePPOPolicy`
"Starting nightly retrain" / "retrain complete" log lines after 21:00
Edmonton, a changed (still in-bounds) action the next morning, and clean
hourly `:33` checkpoints throughout the night.

## Deployment (manual, not automated)

Point the `command:` lines of compose services `zone1`–`zone5` and `zone11` at
these configs and `docker compose up -d zone1 zone2 zone3 zone4 zone5 zone11`
once P0 hands off.
