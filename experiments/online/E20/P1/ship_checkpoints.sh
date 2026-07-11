#!/bin/bash
# Stage the E20/P1 frozen checkpoints + offline replay buffers from aurora
# (where model-uncertainty-exploration trains) into this repo's gitignored
# checkpoints/ tree. Run ON THE DEPLOY HOST from the plant-rl repo root:
#
#   ./experiments/online/E20/P1/ship_checkpoints.sh
#
# On aurora itself the rsync source host prefix is unnecessary; from archcraft
# prefix the sources with aurora: (or run the script there and push instead).

set -euo pipefail

MUE=~/Github/model-uncertainty-exploration
ANALYTIC_RUN="$MUE/runs/20260710/plant_plant-data_visu-v2-v27/gamma0.8_enn_ln/seed_0"
MASKED_RUN="$MUE/runs/20260711/plant_plant-data_visu-v2-v27/gamma0.8_enn_ln_masked_timeobs/seed_0"

mkdir -p checkpoints/E20/P1/{analytic,masked}

rsync -av "$ANALYTIC_RUN/checkpoint/" checkpoints/E20/P1/analytic/checkpoint/
rsync -av "$MASKED_RUN/checkpoint/" checkpoints/E20/P1/masked/checkpoint/

# Offline replay buffers for the AdaptivePPOPolicy zones (Z5/Z11); regenerate
# with model-uncertainty-exploration/scripts/export_offline_transitions.py.
(
    cd "$MUE"
    uv run python scripts/export_offline_transitions.py \
        --reward-mode analytic \
        --out "$OLDPWD/checkpoints/E20/P1/analytic/offline_transitions.npz"
    uv run python scripts/export_offline_transitions.py \
        --reward-mode masked \
        --out "$OLDPWD/checkpoints/E20/P1/masked/offline_transitions.npz"
)

echo "Staged E20/P1 checkpoints:"
find checkpoints/E20/P1 -maxdepth 3 | sort
