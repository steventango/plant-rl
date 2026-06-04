# Experiment 1 — Iteration 1: Schedule Configs

These schedules are designed to systematically explore blue (`-1`) and red (`+1`) recipes at different time scales within a grow cycle. (The largest time scales, 2 wks and 1 wk, have been done in plant-rl E17.)

## Design

Each schedule specifies `action_days` — the days within a cycle on which an intensity decision is made — and `action_inputs` — whether to apply blue (`-1`) or red (`+1`) on that day.

| Schedule | Action Days | Action Inputs 
| Schedule6 | [1, 5, 10] | [-1, -1, +1] |
| Schedule7 | [1, 5, 10] | [-1, +1, -1] |
| Schedule8 | [1, 5, 10] | [-1, +1, +1] |
| Schedule9 | [1, 5, 10] | [+1, -1, -1] | 
| Schedule10 | [1, 5, 10] | [+1, -1, +1] | 

**Note:** 
- PPFD is fixed at 105 during the experiment. 
- No twilight (though plant-rl E17 had twilights, and morning images are taken at the end of twilight at 9:30am)
-----------------------------------------
June 3: 
Network issue caused severe interuption to the experiments. No camera images were taken between Jun3 1am to 1pm. Zones 6, 7 lighting was fine. Zones 9, 10 lighting were delayed by 1-2 hours. Zone 8 was completely off. To salvage the situation the experiments are redesigned as follows. 
- zone 8 is now all blue, to be compared with plant-rl E17's all blue to see (1) the effect of missing a day of light (2) any other differences caused by other changes (mainly lack of twilight)
- zone 7 is switched to -1, +1, +1, to replace zone 8
- zone 10 is switched to +1, +1, -1 so that the remaining 4 zones all test a single lighting change, placed at different days (instead of two lighting changes during the run)

New scheulde: 
| Schedule | Action Days | Action Inputs 
| Schedule6 | [1, 5, 10] | [-1, -1, +1] |
| Schedule7 | [1, 5, 10] | [-1, +1, +1] |
| Schedule8 | [1 ] | [-1 ] |
| Schedule9 | [1, 5, 10] | [+1, -1, -1] | 
| Schedule10 | [1, 5, 10] | [+1, +1, -1] | 
