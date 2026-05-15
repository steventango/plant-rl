# Experiment E18 / Phase P0 — Constant 105 PPFD incubation

## Overview

5-day **post-transplant incubation** for the Trial 17 cohort: constant 105 PPFD balanced-white on the same 18:00 → 06:00 night-shifted photoperiod the [P1 agent-controlled phase](../P1/README.md) will use. All three zones (alliance-zone01, 02, 03) receive the identical policy so the plants enter P1 partially acclimated to 105 PPFD with no spectrum cross-contamination across treatment arms.

This phase runs from **transplant date (DAS 7)** to **agent start (DAS 12)**, after which control hands off to [E18/P1](../P1/README.md). The configs include a 9-day buffer (`total_steps: 14400`) so the run won't terminate prematurely if the handoff is delayed.

## Configs

| Zone | Config | Agent | Wrapper settings |
|---|---|---|---|
| Z1 (zone01) | `Constant1.json` | `ConstantAgent` (`constant_action: 1.0`) | `flash_photography: true`, `enforce_night: true`, `timezone: "Etc/GMT-9"` |
| Z2 (zone02) | `Constant2.json` | same | same |
| Z3 (zone03) | `Constant3.json` | same | same |

All three resolve through `algorithms/registry.py`'s `startswith("Constant")` rule to `ConstantAgent`. `constant_action: 1.0` multiplies by `BALANCED_ACTION_105` in `PlantGrowthChamberIntensity` to emit 105 PPFD across the balanced 5-channel spectrum.

## Photoperiod & flash

Inherits the same flash-photography wrapper mode used in P1 (see `src/algorithms/PlantGrowthChamberAsyncAgentWrapper.py:maybe_enforce_flash_photography_action`):
- 09:00 – 20:59 wrapper-local (chamber 18:00 – 05:59 MDT): daytime, 105 PPFD applied.
- 08:59 wrapper-local (chamber 17:59 MDT): 1-min `BALANCED_ACTION_105` flash for daily camera capture under a standardized spectrum.
- 21:00 – 08:58 wrapper-local: night, action zeroed.

The flash gives the CV pipeline one daily standardized image during incubation too, so plant-area growth is tracked from the beginning of the chamber-controlled period — important for establishing the day-0 baseline before P1's treatment arms diverge.

## Deployment

```bash
python src/main_real.py -e "experiments/online/E18/P0/Constant1.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E18/P0/Constant2.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E18/P0/Constant3.json" -i 0 --deploy
```

Each zone is deployed independently. After the 5-day incubation completes (or earlier, if you're ready to hand off), stop the P0 runs and start the corresponding P1 configs.

## Energy

At constant 105 PPFD over the 12 h photoperiod, lights-on plug power = 51.4 W (per the [E18/P0.1](../P0.1/README.md) pooled fit). Per-zone daily energy ≈ 617 Wh × 5 days ≈ 3 085 Wh for the planned incubation window. Same for all three zones.

## See also

- [`../P0.1/README.md`](../P0.1/README.md) — power characterization sweep, source of the `P(PPFD) = 9.71 + 0.164·PPFD^1.19` fit.
- [`../P0.2/README.md`](../P0.2/README.md) — compressed 12-hour real-hardware test of the P1 schedules.
- [`../P1/README.md`](../P1/README.md) — the 14-day agent-controlled phase that follows incubation.
