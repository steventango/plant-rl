# Experiment E17 / Phase P0

Keep seedlings alive with constant white light using a 12h photoperiod (evening schedule).

## Schedule (Etc/GMT-9 / UTC+9)

| Period | Local time | MDT (UTC-6) |
|--------|-----------|-------------|
| Twilight (dawn) | 09:01–09:29 | 18:01–18:29 |
| Day (constant white) | 09:30–20:29 | 18:30–05:29 |
| Twilight (dusk) | 20:30–20:59 | 05:30–05:59 |
| Night | 21:00–09:00 | 06:00–18:00 |

## Environment
### TemporalPlantGrowthChamberColorTriangle
#### State
  - time

#### Actions
  - color triangle simplex (3D → 6D via BALANCED/RED/BLUE basis)

#### Time step
10 minute

## Agents
### Constant white
`constant_action: [0, 1, 0]` → BALANCED_ACTION_105 (white spectrum at 105 µmol/m²/s)

Twilight ramps enforced by `PlantGrowthChamberAsyncAgentWrapper`.

## Deployment
Using some alliance chambers (zones 1-12)
