# Experiment 1 — Iteration 1: Schedule Configs

These schedules are designed to systematically explore blue (`-1`) and red (`+1`) recipes at different time scales within a grow cycle. (The largest time scales, 2 wks and 1 wk, have been done in plant-rl E17.)

## Design

Each schedule specifies `action_days` — the days within a cycle on which an intensity decision is made — and `action_inputs` — whether to apply blue (`-1`) or red (`+1`) on that day.

| Schedule | Action Days | Action Inputs 
| Schedule6 | [1, 6, 11] | [-1, -1, +1] |
| Schedule7 | [1, 6, 11] | [-1, +1, -1] |
| Schedule8 | [1, 6, 11] | [-1, +1, +1] |
| Schedule9 | [1, 6, 11] | [+1, -1, -1] | 
| Schedule10 | [1, 6, 11] | [+1, -1, +1] | 

**Note:** 
- PPFD is fixed at 105 during the experiment. 
- No twilight (though plant-rl E17 had twilights, and morning images are taken at the end of twilight at 9:30am)