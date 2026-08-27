#!/bin/bash
# Stage the E22/P1 frozen checkpoints + offline replay buffers from the machine
# where model-uncertainty-exploration trains (aurora) into this repo's
# gitignored checkpoints/ tree. Run from the plant-rl repo root:
#
#   ./experiments/online/E22/P1/ship_checkpoints.sh
#
# This script is committed as documentation of the staging step; it stages
# LOCALLY only. Copying to the deploy host and any compose.yml change are done
# manually — archcraft runs live experiments and must not be disturbed here.

set -euo pipefail

MUE=~/Github/model-uncertainty-exploration
RUNS="$MUE/runs/20260827"
LBL=gamma0.8_enn_ln_diffsetup

STANDARD_RUN="$RUNS/plant_plant-data_e18e21-daily-v29/${LBL}_analytic/seed_0"
MASKED_E21_RUN="$RUNS/plant_plant-data_e18e21-daily-v29/${LBL}_masked_log_e21/seed_0"
DIFFERENTIAL_RUN="$RUNS/plant_plant-data_e18e21-daily-diff-v29/${LBL}_analytic_diff/seed_0"
ALL_DATA_RUN="$RUNS/plant_plant-data_all-daily-v29/${LBL}_analytic/seed_0"

mkdir -p checkpoints/E22/P1/{standard,differential,masked_e21,all_data}

rsync -av --delete "$STANDARD_RUN/checkpoint/"     checkpoints/E22/P1/standard/checkpoint/
rsync -av --delete "$DIFFERENTIAL_RUN/checkpoint/" checkpoints/E22/P1/differential/checkpoint/
rsync -av --delete "$MASKED_E21_RUN/checkpoint/"   checkpoints/E22/P1/masked_e21/checkpoint/
rsync -av --delete "$ALL_DATA_RUN/checkpoint/"     checkpoints/E22/P1/all_data/checkpoint/

# Offline replay buffer for the one AdaptivePPOPolicy explore arm (Z4). It also
# carries the dynamics target and the per-day control tables, which the
# differential setup needs and the deploy host cannot derive (no minari, no
# parquet). Z1/Z3/Z5 are frozen exploit policies and need no buffer.
(
    cd "$MUE"
    uv run python scripts/export_offline_transitions.py \
        --dataset plant-data/e18e21-daily-diff-v29 --reward-mode analytic_diff \
        --out "$OLDPWD/checkpoints/E22/P1/differential/offline_transitions.npz"
)

echo "Staged E22/P1 checkpoints:"
find checkpoints/E22/P1 -maxdepth 2 | sort
