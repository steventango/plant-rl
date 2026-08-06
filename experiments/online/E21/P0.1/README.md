# Experiment E21 / Phase P0.1 — Zone 12 per-channel power sweep

Collect solo-channel intensity vs smart-plug power on alliance-zone12 so each LED channel can get its own power fit (unlike [E18/P0.1](../../E18/P0.1/README.md), which ramped balanced white as a single scalar).

## Environment

`PlantGrowthChamber` with `action: "ppfd6"`. The wrapper re-polls the agent every `action_timestep: 5` minutes so each 6-vector is held for one smart-plug / CSV cycle. Night/twilight and flash photography are off (`enforce_night: false`) so the sweep runs continuously. CV stays disabled (`enable_cv_pipeline: false`).

### Cadence

- env step duration: 1 min
- agent re-poll (`action_timestep`): 5 min
- smart-plug / CSV writer gates: on `minute % 5 == 0`

### Actions

Six-channel PPFD vectors. Only one channel is non-zero at a time.

## Safe ranges (`configs/calibration.json`)

| Channel | Index | safe_minimum | safe_maximum |
|---|---|---|---|
| blue | 0 | 5 | 96 |
| cool_white | 1 | 5 | 90 |
| warm_white | 2 | 5 | 65 |
| orange_red | 3 | 4 | 79 |
| red | 4 | 5 | 55 |
| far_red | 5 | ≈0.668 | ≈21.636 |

## Sweep schedule

For each channel `c` in order blue → far_red: 21 ascending levels

`ℓ_i = i × safe_maximum[c] / 20` for `i = 0..20`

with action `[0,…,ℓ_i,…,0]` (only index `c` set). Then a final all-off vector.

- 6 × 21 + 1 = **127** schedule steps
- each held 5 min → **~10.6 h** (`total_steps: 635`)

Levels below `safe_minimum` are zeroed by calibration (duplicate near-zero power samples at the bottom of each channel ramp), same intentional trade-off as E18/P0.1.

## Agent

`SequenceChannelSweep12` → `SequenceAgent`. `actions` is a JSON-stringified list of 127 six-vectors.

## Deployment

```bash
# compose.yml zone12 already points here; recreate to pick it up:
docker compose up -d --force-recreate zone12

# or run directly:
python src/main_real.py -e "experiments/online/E21/P0.1/ChannelSweep12.json" -i 0 --deploy
```

Power data lands in the zone’s `raw_YYYY-MM-DD.csv` (`power` / `voltage` / `current` alongside `action.*`). After the run, fit per channel by grouping rows on the active channel index and regressing channel PPFD vs `power` — adapt [`../../E18/P0.1/analyze_power.py`](../../E18/P0.1/analyze_power.py).

Incubation config for handback: [`../P0/Constant12.json`](../P0/Constant12.json).
