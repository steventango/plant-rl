# Experiment E18 / Phase P0.2 — 12-hour real-hardware test of the P1 schedules

## Overview

**Pre-deploy shakedown.** Compresses the [E18/P1](../P1/README.md) schedules by ~48× so the entire Z1 / Z2 / Z3 three-arm experiment plays in **3.5 h** (210 env steps), letting us validate the full deploy pipeline — registry resolution, wrapper night/twilight/flash handling, lightbar response, smart-plug telemetry, CV pipeline, CSV writes — end-to-end on the real hardware in less than half a work shift, before committing to the 14-day live trial. All three zones run within the chamber's evening 12 h photoperiod (chamber 18:00 → 21:30 MDT covers the 3.5 h test if started at 18:00).

All `action_timestep` values are multiples of 5 min to align with the chamber's 5-min CSV writer / smart-plug telemetry cadence (so action transitions land on telemetry samples instead of between them). Z1 plays its full 14-step power-law ramp once; Z2 plays 14 compressed parabolic cycles; Z3 stays constant. The 3.5 h cutoff is forced by `total_steps: 210` on every config — Z1's natural completion time — so no zone clamps at its last entry.

## Configs

| Zone | Config | Shape | `action_timestep` (min) | Entries | Total runtime |
|---|---|---|---|---|---|
| **Z1 (zone01)** | `SequencePowerLawRamp1.json` | 14-step power-law ramp (40 → 130 PPFD) | **15** | 14 | 14 × 15 = 210 min |
| **Z2 (zone02)** | `SequenceParabolic2.json` | parabola `[60, 126, 60]` × 14 cycles | **5** | 42 | 42 × 5 = 210 min |
| **Z3 (zone03)** | `Constant3.json` | constant 105 PPFD | 720 | — (constant) | 210 min |

`total_steps: 210` on every config — exactly 3.5 h at 1-min env steps. The deploy stops at Z1's natural ramp completion; Z2 has cycled 14 parabolas; Z3 has held constant the whole time. No zone clamps at its last entry.

Energy ratios match P1 (verified analytically against `P(PPFD) = 9.71 + 0.164·PPFD^1.19`):

| Zone | Energy over 3.5 h (Wh) | Ratio vs Z3 |
|---|---|---|
| Z1 P0.2 | 144.2 | **80.15 %** |
| Z2 P0.2 | 144.4 | **80.26 %** |
| Z3 P0.2 | 179.9 | 100 % |

## What this test exercises

- **Registry resolution.** All three agent names (`SequencePowerLawRamp1`, `SequenceParabolic2`, `Constant3`) must hit the right classes via `algorithms/registry.py`.
- **Wrapper polling cadence.** Z1's 15-min `action_timestep` should fire 14 polls in 3.5 h; Z2's 5-min should fire 42 polls; Z3 is constant.
- **Multi-channel actuation.** Z1 spans 40 → 130 PPFD so all four LED channels (blue, cool_white, warm_white, red) activate across the run as the schedule crosses the calibration safe_min thresholds (warm_white at PPFD ≥ 67, red at PPFD ≥ 85). Z2 cycles between blue+cool_white-only (60 PPFD edges) and full balanced (126 PPFD peak) every 5 min. Z3 stays full balanced throughout.
- **Smart-plug telemetry.** Each zone's recorded `power` should track its scheduled PPFD via the [E18/P0.1](../P0.1/README.md) power-law fit. Diagnostic signature: Z1 ramping, Z2 oscillating 60↔126, Z3 flat at 51.4 W.
- **CV flash photography.** Wrapper-local 08:59 (chamber 17:59 MDT) flash. With a 3.5 h test starting at chamber 18:00, the flash fires *just before* the test (1 min earlier). To exercise the flash within P0.2 itself, deploy at chamber 17:59 — then the run starts with the 1-min flash, transitions to daytime at 18:00, and the schedules begin. Otherwise the flash logic is already validated by `test_via_main_real.py` and the 14-day simulation under [`../P1/figures/`](../P1/figures/).
- **Twilight off.** All configs set `flash_photography: true`, which collapses dawn/dusk to a hard square wave plus the 1-min flash. Confirm zero PPFD outside the daytime window.

## Deployment

```bash
# Z1: compressed power-law ramp
python src/main_real.py -e "experiments/online/E18/P0.2/SequencePowerLawRamp1.json" -i 0 --deploy

# Z2: compressed within-day parabola
python src/main_real.py -e "experiments/online/E18/P0.2/SequenceParabolic2.json" -i 0 --deploy

# Z3: constant 105 control
python src/main_real.py -e "experiments/online/E18/P0.2/Constant3.json" -i 0 --deploy
```

Start all three simultaneously at chamber wall-clock 18:00 MDT. All three configs use `total_steps: 210` (3.5 h at 1-min env steps), so the run completes around chamber 21:30 MDT — well within the 12 h photoperiod.

## Verification

Post-deploy:
1. `rsync` raw CSVs from `archcraft:/data/plant-rl/online/E18/P0.2/...`
2. Plot `action.*` and `power` per zone — Z1 should show 14 stepped levels, Z2 should show ~42 oscillations, Z3 should be flat.
3. Integrate per-zone energy over the 11.9 h active window; confirm ratios match the table above to within ~3 % (the Kasa KP125M factory floor — see [`../P1/README.md`](../P1/README.md) sensor-noise section).
4. If anything diverges from these expected signatures, **do not proceed to P1**. Diagnose the deviation against the simulator outputs in [`../P1/figures/`](../P1/figures/) (the same code paths exercised at 14-day scale).

## See also

- [`../P0/README.md`](../P0/README.md) — preceding 5-day incubation under constant 105 PPFD.
- [`../P0.1/README.md`](../P0.1/README.md) — power characterization sweep that gave us `P(PPFD)`.
- [`../P1/README.md`](../P1/README.md) — the full 14-day deploy this phase shakes down.
- [`../P1/test_via_main_real.py`](../P1/test_via_main_real.py) — same configs run through the `main_real` integration path on the v27 mock dataset.
