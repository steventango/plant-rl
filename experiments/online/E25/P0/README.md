# Experiment E25 / Phase P0 — Constant 100 PPFD incubation (Z12)

## Overview

Post-transplant incubation on alliance-zone12: constant 100 PPFD balanced-white on the same night-shifted photoperiod the following agent-controlled phase will use. This phase covers zone 12 only, so that chamber enters the next phase acclimated to 100 PPFD.

This phase runs from **transplant date (2026-11-19, DAS 7)** to **agent start (2026-11-25, DAS 12)** — 6 days of incubation. `total_steps: -1` so the constant-100 policy does not expire if the handoff is delayed.

Follows **E24/P0** `Constant12.json` on the same zone and the same policy (`enable_cv_pipeline: false`).

## Configs

| Zone | Config | Agent | Wrapper settings |
|---|---|---|---|
| Z12 (zone12) | `Constant12.json` | `ConstantAgent` (`constant_action: 1.0`) | `flash_photography: true`, `enforce_night: true`, `timezone: "Etc/GMT-2"`, `enable_cv_pipeline: false` |

`Constant12` resolves through `algorithms/registry.py`'s `startswith("Constant")` rule to `ConstantAgent`. `constant_action: 1.0` multiplies by `BALANCED_ACTION_100` (the `PlantGrowthChamberIntensity` reference spectrum) to emit 100 PPFD across the balanced 5-channel spectrum.

## Photoperiod & flash

Uses the flash-photography wrapper mode (see `src/algorithms/PlantGrowthChamberAsyncAgentWrapper.py:maybe_enforce_flash_photography_action`):
- 09:00 – 20:59 wrapper-local: daytime, 100 PPFD applied.
- 08:59 wrapper-local: 1-min `BALANCED_ACTION_40` flash (40 PPFD on the balanced 5-channel spectrum) for daily camera capture under a standardized spectrum.
- 21:00 – 08:58 wrapper-local: night, action zeroed.

The flash gives the CV pipeline one daily standardized image during incubation too, so plant-area growth is tracked from the beginning of the chamber-controlled period — important for establishing the day-0 baseline before the agent-controlled phase's treatment arms diverge.

## Deployment

```bash
python src/main_real.py -e "experiments/online/E25/P0/Constant12.json" -i 0 --deploy
```

Zone 12 is deployed via `compose.yml` (`docker compose up -d zone12`). After the 6-day incubation completes (or earlier, if ready to hand off), stop the P0 run and start the corresponding P1 config once E25/P1 is created.

## Energy

At constant 100 PPFD over the 12 h photoperiod, lights-on plug power ≈ 49.05 W (per `P(PPFD) = 9.71 + 0.164·PPFD^1.19`, the pooled fit from [E18/P0.1](../../E18/P0.1/README.md)). Daily energy ≈ 589 Wh; over the 6-day planned-incubation window that's ≈ 3534 Wh for zone 12.

## See also

- [`../../E18/P0.1/README.md`](../../E18/P0.1/README.md) — power characterization sweep, source of the `P(PPFD) = 9.71 + 0.164·PPFD^1.19` fit.
- [`../../E23/P0/README.md`](../../E23/P0/README.md) — prior full-chamber incubation phase.
- [`../../E24/P0/Constant12.json`](../../E24/P0/Constant12.json) — zone 12 policy this phase continues.
