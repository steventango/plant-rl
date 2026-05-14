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
