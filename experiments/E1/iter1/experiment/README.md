# Experiment 1 — Iteration 1: Schedule Configs

These schedules are designed to systematically explore minimum (`-1`) and maximum (`+1`) light intensity at different time scales within a grow cycle.

## Design

Each schedule specifies `action_days` — the days within a cycle on which an intensity decision is made — and `action_inputs` — whether to apply minimum (`-1`) or maximum (`+1`) PPFD on that day.

| Schedule | Action Days | Action Inputs | Description |
|----------|-------------|---------------|-------------|
| Schedule4 | [1] | [-1] | Constant min intensity |
| Schedule5 | [1] | [+1] | Constant max intensity |
| Schedule6 | [1, 8] | [-1, +1] | Start low, switch to high mid-cycle |
| Schedule7 | [1, 8] | [+1, -1] | Start high, switch to low mid-cycle |
| Schedule8 | [1, 6, 11] | [-1, -1, +1] | Low early, ramp to high near end |
| Schedule9 | [1, 6, 11] | [-1, +1, -1] | Low-high-low pattern |
| Schedule10 | [1, 6, 11] | [-1, +1, +1] | Start low, stay high |
| Schedule11 | [1, 6, 11] | [+1, -1, -1] | Start high, stay low |
| Schedule12 | [1, 6, 11] | [+1, -1, +1] | High-low-high pattern |

**Note:** `[+1, +1, -1]` was excluded because only zones 4-12 are available and each schedule is assigned to one zone. 
