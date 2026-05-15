# Experiment E18 / Phase P0.1

Power characterization sweep. Drive a deterministic intensity ramp on the balanced white spectrum from 0 to safe-max, recording smart-plug power readings at each level. Run on zones 1 and 2, two repeats per zone.

## Environment
### PlantGrowthChamberIntensity
The problem wraps the agent with `PlantGrowthChamberAsyncAgentWrapper`. The wrapper re-polls the agent on a configurable `action_timestep` (here: 5 min) so each scalar from the agent is held steady for one full smart-plug / CSV cycle. Night/twilight enforcement is disabled (`enforce_night: false`) so the sweep runs continuously regardless of wall-clock time.

#### State
  - time

#### Actions
  - scalar `s` in `[0, s_max]` → `s × BALANCED_ACTION_105` (6-channel PPFD)

#### Cadence
- env step duration: 1 min (default)
- agent re-poll (`action_timestep`): 5 min
- smart-plug / CSV writer gates: on `minute % 5 == 0`

So the agent emits a fresh intensity every 5 min; the env applies that same intensity for 5 × 1-min steps; one CSV row is written per 5-min boundary, with action and power aligned.

## Safe-max derivation

`BALANCED_ACTION_105 = [blue=19.5, cool_white=71.53, warm_white=7.82, orange_red=0, red=6.15, far_red=0]` (105 PPFD total).

Per-channel `safe_maximum` (`configs/calibration.json`): blue=96, cool_white=90, warm_white=65, orange_red=79, red=55, far_red=21.6.

Cool_white is the limiting channel: `s_max = 90 / 71.53 ≈ 1.258213`. At s_max the action is `[24.53, 90.00, 9.84, 0, 7.74, 0]` ≈ 132.1 PPFD total. No per-channel clipping fires at this peak.

## Safe-minimum note (spectrum at low levels)

`Calibration.get_calibrated_action` gates each channel by its per-channel `safe_minimum` (`configs/calibration.json`: blue=5, cool_white=5, warm_white=5, red=5). Below that PPFD the channel is zeroed and the remaining active channels are rescaled to hit the target PPFD. With the balanced-105 shares, channels activate at:

- cool_white: `s ≥ 5/71.53 ≈ 0.070` (first hit at s₂ = 0.1258)
- blue: `s ≥ 5/19.50 ≈ 0.256` (first hit at s₅ = 0.3146)
- warm_white: `s ≥ 5/7.82 ≈ 0.639` (first hit at s₁₁ = 0.6920)
- red: `s ≥ 5/6.15 ≈ 0.813` (first hit at s₁₃ = 0.8178)

So below s ≈ 0.813 the emitted spectrum is **not** the balanced 105 reference — it is cool_white-dominant at the bottom, gradually adding blue, then warm_white, then red. The trade is intentional: we accept spectrum distortion at low intensities to retain power measurements near 0 PPFD, which we couldn't otherwise sample. Interpret the low-end rows of the CSV as power for the active sub-spectrum, not for balanced-105.

Note also: s₁ = 0.0629 falls below every channel's threshold, so it emits 0 PPFD (a duplicate of s₀).

## Sweep schedule

21 ascending levels at 5% increments of `s_max`, `s_i = i × s_max / 20` for `i = 0..20`, played twice back-to-back (42 distinct intensity levels). Each level holds for 5 min → ~3.5 h per zone. With 1-min env steps that is `42 × 5 = 210` env steps (`total_steps`).

## Agents
### SequenceAgent
`actions` (JSON-stringified list of 42 scalars) is consumed by `SequenceAgent`. The wrapper rate-limits agent polls to one per 5-min `action_timestep`. The `PlantGrowthChamberIntensity` env multiplies each scalar by the reference spectrum.

## Deployment

```bash
python src/main_real.py -e "experiments/online/E18/P0.1/IntensitySweep1.json" -i 0 --deploy
python src/main_real.py -e "experiments/online/E18/P0.1/IntensitySweep2.json" -i 0 --deploy
```

Power data is written to the standard `raw_YYYY-MM-DD.csv` files under each zone's data directory (`power`, `voltage`, `current` columns alongside the action vector).
