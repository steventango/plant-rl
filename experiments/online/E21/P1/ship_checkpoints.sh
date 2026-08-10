#!/bin/bash
# Stage the E21/P1 frozen checkpoints + offline replay buffers from the machine
# where model-uncertainty-exploration trains (aurora) into this repo's
# gitignored checkpoints/ tree. Run from the plant-rl repo root:
#
#   ./experiments/online/E21/P1/ship_checkpoints.sh
#
# From another host, prefix the rsync sources with aurora: (or run this on
# aurora and rsync the resulting checkpoints/E21 tree across).

set -euo pipefail

MUE=~/Github/model-uncertainty-exploration
ANALYTIC_RUN="$MUE/runs/20260807/plant_plant-data_visu-v28/gamma0.8_enn_ln/seed_0"
MASKED_LOG_RUN="$MUE/runs/20260731/plant_plant-data_visu-v28/gamma0.8_enn_ln_masked_log/seed_0"

mkdir -p checkpoints/E21/P1/{analytic,masked_log}

rsync -av --delete "$ANALYTIC_RUN/checkpoint/" checkpoints/E21/P1/analytic/checkpoint/
rsync -av --delete "$MASKED_LOG_RUN/checkpoint/" checkpoints/E21/P1/masked_log/checkpoint/

# Offline replay buffers for the AdaptivePPOPolicy explore arms (Z3/Z4).
# masked_log exports [log_area, time]; analytic exports [log_area].
(
    cd "$MUE"
    uv run python scripts/export_offline_transitions.py \
        --dataset plant-data/visu-v28 --reward-mode analytic \
        --out "$OLDPWD/checkpoints/E21/P1/analytic/offline_transitions.npz"
    uv run python scripts/export_offline_transitions.py \
        --dataset plant-data/visu-v28 --reward-mode masked_log \
        --out "$OLDPWD/checkpoints/E21/P1/masked_log/offline_transitions.npz"
)

echo "Staged E21/P1 checkpoints:"
find checkpoints/E21/P1 -maxdepth 2 | sort
