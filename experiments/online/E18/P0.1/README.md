# Experiment E18 / Phase P0.1

Power characterization sweep. Drive a deterministic intensity ramp on the balanced white spectrum from 0 to safe-max, recording smart-plug power readings at each level. Run on zones 1 and 2, two repeats per zone.

## Environment
### PlantGrowthChamberIntensity
#### State
  - time

#### Actions
  - scalar `s` in `[0, s_max]` → `s × BALANCED_ACTION_105` (6-channel PPFD)

#### Time step
5 minutes (aligned with the smart-plug fetch / CSV writer gate)

No twilight ramps (uses `BaseAsyncProblem` / `AsyncAgentWrapper`).

## Safe-max derivation

`BALANCED_ACTION_105 = [blue=19.5, cool_white=71.53, warm_white=7.82, orange_red=0, red=6.15, far_red=0]` (105 PPFD total).

Per-channel `safe_maximum` (`configs/calibration.json`): blue=96, cool_white=90, warm_white=65, orange_red=79, red=55, far_red=21.6.

Cool_white is the limiting channel: `s_max = 90 / 71.53 ≈ 1.258213`. At s_max the action is `[24.53, 90.00, 9.84, 0, 7.74, 0]` ≈ 132.1 PPFD total. No per-channel clipping fires at this peak.

## Sweep schedule

21 ascending levels at 5% increments of `s_max`, `s_i = i × s_max / 20` for `i = 0..20`, played twice back-to-back (42 steps total, ≈3.5 h per zone).

## Agents
### SequenceAgent
`actions` (JSON-stringified list of 42 scalars) is consumed by `SequenceAgent`. The `PlantGrowthChamberIntensity` env multiplies each scalar by the reference spectrum.

## Deployment

```bash
python src/main_real.py -e "experiments/online/E18/P0.1/IntensitySweep1.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E18/P0.1/IntensitySweep2.json" -i 0 --deploy
```

Power data is written to the standard `raw_YYYY-MM-DD.csv` files under each zone's data directory (`power`, `voltage`, `current` columns alongside the action vector).
