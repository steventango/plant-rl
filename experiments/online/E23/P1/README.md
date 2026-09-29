# Experiment E23 / Phase P1 — E22/P1 rerun on E18+E21+E22 data, plus a constant alternative spectrum

## Status: staged locally, NOT deployed

Configs, checkpoints and tests are prepared on aurora only. Nothing has been
copied to the deploy host and `compose.yml` is untouched.

Follows **E23/P0** (constant-100 incubation, transplant 2026-09-23, planned agent
start 2026-09-29). `Constant11.json` matches `E23/P0/Constant11.json` except for
`enable_cv_pipeline: true`, so Z11 continues its P0 policy into P1.

## What changed from E22/P1

The design is [E22/P1](../../E22/P1/README.md)'s: same arms, same hyperparameters
(`gamma 0.8`, ENN + model LayerNorm, 10 seeds, `obs_dim 2`), same ops conventions.
Read that README for how the differential setup and the masked gate work, and for
the caveats. Three things differ:

1. **Training data now includes E22.** E22 has its own constant-white Z11, so it
   joins the differential set. Datasets come from plant-data `generate_e23.sh`:
   `plant-data/e18e21e22-daily-v30` (standard, masked) and
   `plant-data/e18e21e22-daily-diff-v30` (differential). Both come from one parquet
   (E18 + E21 + E22 with A2/A3, 22,003 transitions). Every row has an in-batch
   control, so the differential arm also trains on all 22,003.
2. **E22 Z3 is truncated at day 10.** One of its two trays (plants 32–63) was lost
   between the 2026-09-10 and 2026-09-11 captures. Days 0–10 have all 62 plants,
   day 11 has 38, and from day 13 only 30 remain. Rows from day 11 on are dropped
   (`config.ZONE_DAY_CUTOFF` in plant-data), so the zone's episode ends cleanly at
   day 10 rather than stepping into a survivor-only mean.
3. **Z5 is a constant blue + orange-red spectrum**, replacing E22's `all_data` arm.

## Configs

| Zone | Config | Agent | Checkpoint | Head | Selection | Nightly retrain |
|---|---|---|---|---|---|---|
| Z1 | `Z1.json` | `PPOPolicy1` | standard | `ppo_eval` | mean | — |
| Z2 | `Z2.json` | `PPOPolicy2` | differential | `ppo_eval` | mean | — |
| Z3 | `Z3.json` | `PPOPolicy3` | masked_e21 | `ppo_eval` | mean | — |
| Z4 | `Z4.json` | `AdaptivePPOPolicy4` | differential | `ppo_explore` | sampled | `analytic_diff` |
| Z5 | `Constant5.json` | `ConstantAgent` | — (blue + orange-red, 100 PPFD) | — | — | — |
| Z11 | `Constant11.json` | `ConstantAgent` | — (constant white) | — | — | — |

Checkpoints (`runs/20260928/` in model-uncertainty-exploration, seed 0):

- standard — `plant_plant-data_e18e21e22-daily-v30/gamma0.8_enn_ln_diffsetup_analytic` (`--include-time`)
- differential — `plant_plant-data_e18e21e22-daily-diff-v30/gamma0.8_enn_ln_diffsetup_analytic_diff`
- masked_e21 — `plant_plant-data_e18e21e22-daily-v30/gamma0.8_enn_ln_diffsetup_masked_log_e21`

Zones 6–10 and 12 are not part of this phase.

## Z3: the masked gate is still pinned to E21 Z11

`masked_log_e21` is reused unchanged. Rerunning `scripts/compute_masked_baseline.py`
on the new parquet (E21 Z11) reproduces its baseline and surplus constants
**bit-for-bit**: adding E22 to the join does not move any E21 Z11 rows. Only the
off-target fraction changes with the data, to 46.7% of transitions (48.1% on v29).

The E22 cohort was much smaller than E21's (day-14 Z11 mean area 3.44 vs 5.25), so
this gate is hard to reach for a cohort like E22's. Below the gate the reward pays
for closing the gap, so a small cohort gets **more** light, not less. On the policy
grid, E22 Z11's areas sit right at the gate on days 0 and 7, and clearly below it
by day 14 (log area 1.24 vs a gate of 1.55), where the mean action is 1.30, the
maximum.

The mock run's constant 40 PPFD says nothing about a small cohort. The mock replays
E17 data, which sits above the E21 gate throughout, so the policy correctly coasts
at the floor there.

## Z5: constant blue + orange-red

`action: ppfd6`, `constant_action: [[30.90, 0, 0, 69.10, 0, 0]]`
(blue, cool_white, warm_white, orange_red, red, far_red), for 100 PPFD total.

| spectrum | blue | cool_white | warm_white | orange_red | red | far_red |
|---|---|---|---|---|---|---|
| white (`BALANCED_ACTION_100`) | 18.57 | 68.12 | 7.45 | 0 | 5.86 | 0 |
| Z5 | 30.90 | 0 | 0 | 69.10 | 0 | 0 |

- At 100 PPFD it draws 46.96 W, against 48.23 W for white.
- It delivers about 5.6% more yield photon flux per watt (YPF/W).
- Its blue-photon share (400–500 nm) matches white exactly, at 30.9%.
- Orange-red at 69.1 is under its 79 PPFD safe maximum.

Because blue is held equal, a rosette-area difference against Z11 can be put down
to swapping white for orange-red, not to the plants getting less blue. The better
YPF/W is a design property; that it grows bigger rosettes is what this zone tests.

The vector is wrapped in an extra list on purpose. The experiment loader treats a
bare list in `metaParameters` as a **hyperparameter sweep**, so
`[30.90, 0, 0, 69.10, 0, 0]` would silently run permutation 0 as
`constant_action = 30.9`, i.e. 30.9× white, about 3090 PPFD.
`test_z5_is_the_constant_orange_red_spectrum` loads the config through the real
loader to catch this.

Flash photography still captures at `BALANCED_ACTION_40` (white), so Z5's daily
image is taken under the same standardized spectrum as every other zone.
`enable_cv_pipeline: true`, as for Z11. The constant agents ignore the CV
output; it is on so both constant zones log hourly plant area alongside the agent
arms. That makes 6 CV zones (Z1–Z5, Z11), fewer than the 7 that plant-cv served
through E22 with every request succeeding (slowest hourly batch finished 25 s
after :00, against the 45 s `CV_REQUEST_TIMEOUT_S`).

## Policy behaviour (mean action, golden grid)

| arm | day 0 | day 7 | day 14 |
|---|---|---|---|
| standard | 0.81 → 1.30, then 0.40 for the largest plants | 0.64 → 1.08, 0.40 at the top | 0.55–0.67, 0.40 at the top |
| differential | 0.60 → 1.30 | 0.40 for small plants, 0.64–0.78 for large | **0.40 everywhere** |
| masked_e21 | 0.83–1.06 below the gate, 0.40 above | 0.68–0.90 below, 0.40 above | 0.65–1.30 below, 0.40 above |

The one qualitative change from E22/P1 is the differential arm at day 14. The
arm now sits on the floor across the whole area range, where E22/P1's version
rose back to about 0.88 for large plants. So it now ends with the floor
saturation E22/P1's README flagged for the standard arm (caveat 5).

Mock-chamber daytime PPFD over 14 simulated days (`test_via_main_real.py`):

| zone | levels | range |
|---|---|---|
| Z1 | 6 | 40 – 115 |
| Z2 | 9 | 40 – 117 |
| Z3 | 1 | 40 (the E17 mock data sits above the gate throughout) |
| Z4 | 2 | 40, 130 (explore head still bang-bang, E22/P1 caveat 1) |
| Z5 | 1 | 100 |
| Z11 | 1 | 100 |

## Staging and tests

```
./experiments/online/E23/P1/ship_checkpoints.sh      # stages checkpoints/E23/P1
JAX_PLATFORMS=cpu uv run pytest tests/algorithms/test_E23P1_parity.py
python experiments/online/E23/P1/test_via_main_real.py
```

Goldens: m-u-e `scripts/dump_golden_actions.py --experiment E23`.

## Deployment

```bash
python src/main_real.py -e "experiments/online/E23/P1/Z1.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E23/P1/Z2.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E23/P1/Z3.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E23/P1/Z4.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E23/P1/Constant5.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E23/P1/Constant11.json" -i 0 --deploy
```

Copy `checkpoints/E23/P1/` to `/app/checkpoints/E23/P1/` on the deploy host first.
