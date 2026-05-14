# Experiment 1 — Iteration 1: Schedule Configs

These schedules are designed to systematically explore minimum (`-1`) and maximum (`+1`) light intensity at different time scales within a grow cycle.

## Design

Each schedule specifies `action_days` — the days within a cycle on which an intensity decision is made — and `action_inputs` — whether to apply minimum (`-1`) or maximum (`+1`) PPFD on that day.

| Schedule | Action Days | Action Inputs | Description |
|----------|-------------|---------------|-------------|
| Schedule1 | [1] | [-1] | Constant min intensity |
| Schedule2 | [1] | [+1] | Constant max intensity |
| Schedule3 | [1, 8] | [-1, +1] | Start low, switch to high mid-cycle |
| Schedule4 | [1, 8] | [+1, -1] | Start high, switch to low mid-cycle |
| Schedule5 | [1, 6, 11] | [-1, -1, +1] | Low early, ramp to high near end |
| Schedule6 | [1, 6, 11] | [-1, +1, -1] | Low-high-low pattern |
| Schedule7 | [1, 6, 11] | [-1, +1, +1] | Start low, stay high |
| Schedule8 | [1, 6, 11] | [+1, -1, -1] | Start high, stay low |
| Schedule9 | [1, 6, 11] | [+1, -1, +1] | High-low-high pattern |

**Note:** Schedule10 (`[+1, +1, -1]`) was excluded because only zones 1–9 are available and each schedule is assigned to one zone. The complementary all-high and all-low cases are already covered by Schedules 1 and 2.
