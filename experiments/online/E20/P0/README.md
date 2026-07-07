# Experiment E20 / Phase P0 — Constant 100 PPFD incubation

## Overview

Post-transplant incubation across all 12 chambers (alliance-zone01 – alliance-zone12): constant 100 PPFD balanced-white on the same night-shifted photoperiod the following agent-controlled phase will use. All twelve zones receive the identical policy so plants enter the next phase partially acclimated to 100 PPFD with no spectrum cross-contamination across treatment arms.

This phase runs from **transplant date (2026-07-07, DAS 7)** to **agent start (2026-07-13, DAS 12)** — 6 days of incubation. The configs are sized for **4 weeks** (`total_steps: 40320` = 28 days of 1-min env steps) so the same constant-100 policy can keep running indefinitely if the handoff is delayed.

## Configs

| Zone | Config | Agent | Wrapper settings |
|---|---|---|---|
| Z1 (zone01) | `Constant1.json` | `ConstantAgent` (`constant_action: 1.0`) | `flash_photography: true`, `enforce_night: true`, `timezone: "Etc/GMT-2"` |
| Z2 (zone02) | `Constant2.json` | same | same |
| Z3 (zone03) | `Constant3.json` | same | same |
| Z4 (zone04) | `Constant4.json` | same | same |
| Z5 (zone05) | `Constant5.json` | same | same |
| Z6 (zone06) | `Constant6.json` | same | same |
| Z7 (zone07) | `Constant7.json` | same | same |
| Z8 (zone08) | `Constant8.json` | same | same |
| Z9 (zone09) | `Constant9.json` | same | same |
| Z10 (zone10) | `Constant10.json` | same | same |
| Z11 (zone11) | `Constant11.json` | same | same |
| Z12 (zone12) | `Constant12.json` | same | same |

All twelve resolve through `algorithms/registry.py`'s `startswith("Constant")` rule to `ConstantAgent`. `constant_action: 1.0` multiplies by `BALANCED_ACTION_100` (the `PlantGrowthChamberIntensity` reference spectrum) to emit 100 PPFD across the balanced 5-channel spectrum.

## Photoperiod & flash

Uses the flash-photography wrapper mode (see `src/algorithms/PlantGrowthChamberAsyncAgentWrapper.py:maybe_enforce_flash_photography_action`):
- 09:00 – 20:59 wrapper-local: daytime, 100 PPFD applied.
- 08:59 wrapper-local: 1-min `BALANCED_ACTION_40` flash (40 PPFD on the balanced 5-channel spectrum) for daily camera capture under a standardized spectrum.
- 21:00 – 08:58 wrapper-local: night, action zeroed.

The flash gives the CV pipeline one daily standardized image during incubation too, so plant-area growth is tracked from the beginning of the chamber-controlled period — important for establishing the day-0 baseline before the agent-controlled phase's treatment arms diverge.

## Deployment

```bash
python src/main_real.py -e "experiments/online/E20/P0/Constant1.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant2.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant3.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant4.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant5.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant6.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant7.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant8.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant9.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant10.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant11.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E20/P0/Constant12.json" -i 0 --deploy
```

Each zone is deployed independently via `compose.yml`. After the 6-day incubation completes (or earlier, if ready to hand off), stop the P0 runs and start the corresponding P1 configs once E20/P1 is created.

## Energy

At constant 100 PPFD over the 12 h photoperiod, lights-on plug power ≈ 49.05 W (per `P(PPFD) = 9.71 + 0.164·PPFD^1.19`, the pooled fit from [E18/P0.1](../../E18/P0.1/README.md)). Per-zone daily energy ≈ 589 Wh; over the 6-day planned-incubation window that's ≈ 3534 Wh per zone. Same for all twelve zones.

## See also

- [`../../E18/P0.1/README.md`](../../E18/P0.1/README.md) — power characterization sweep, source of the `P(PPFD) = 9.71 + 0.164·PPFD^1.19` fit.
- [`../../E18/P0/README.md`](../../E18/P0/README.md) / [`../../E19/P0`](../../E19/P0) — prior incubation phases this one follows.
