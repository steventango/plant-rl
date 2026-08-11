"""Rebuild an AdaptivePPOPolicy retrain archive from an agent state checkpoint.

A night's retrain artifacts are normally archived by
``AdaptivePPOPolicy._archive_retrain`` the moment the new policy is swapped in.
If that archive was skipped (e.g. the target directory was already taken), the
retrained policy is not lost: the hourly agent checkpoint carries the swapped-in
network, its obs normalizer, the updated world model and the collected online
transitions. This reconstructs the same on-disk layout from that checkpoint.

Must run where the agent's configured paths resolve — i.e. inside the zone
container, since configs reference /app/checkpoints and /data/plant-rl:

    docker compose run --rm --no-deps -T zone3 \
        uv run python scripts/salvage_retrain_archive.py \
            --checkpoint checkpoints/results/online/E21/P1/AdaptivePPOPolicy3/0/chk.pkl.xz \
            --out /data/plant-rl/online/E21/P1/AdaptivePPOPolicy3/alliance-zone03/retrain_archive
"""

import argparse
import json
import lzma
import os
import pickle
import sys

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


def _mtime(path: str) -> str:
    from datetime import datetime, timezone

    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _find_agent(obj, depth: int = 0):
    """Locate the AdaptivePPOPolicy inside the pickled checkpoint storage."""
    if depth > 6 or obj is None:
        return None
    if hasattr(obj, "_retrain_count") and hasattr(obj, "_network"):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_agent(v, depth + 1)
            if found is not None:
                return found
    for attr in ("agent", "_agent"):
        if hasattr(obj, attr):
            found = _find_agent(getattr(obj, attr), depth + 1)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="chk.pkl.xz path")
    parser.add_argument("--out", required=True, help="retrain_archive directory")
    parser.add_argument(
        "--label",
        default=None,
        help="archive subdirectory name (default <last_retrain_date>_retrainNNNN_salvaged)",
    )
    args = parser.parse_args()

    with lzma.open(args.checkpoint, "rb") as f:
        storage = pickle.load(f)
    agent = _find_agent(storage)
    if agent is None:
        raise SystemExit("no AdaptivePPOPolicy found in the checkpoint")

    count = agent._retrain_count
    date = getattr(agent, "_last_retrain_date", None)
    if count < 1:
        raise SystemExit(f"checkpoint predates any completed retrain (count={count})")

    label = args.label or f"{date}_retrain{count:04d}_salvaged"
    out = os.path.join(os.path.abspath(args.out), label)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; pass a different --label")
    os.makedirs(out)

    checkpointer = ocp.StandardCheckpointer()
    for name, module in (
        ("network", agent._network),
        ("obs_norm", agent._obs_norm),
        ("model", agent._model),
    ):
        _, state = nnx.split(module)
        checkpointer.save(os.path.join(out, name), state)
    checkpointer.wait_until_finished()

    n, p = agent._offline_count, agent._pointer
    np.savez(
        os.path.join(out, "online_transitions.npz"),
        obs=agent._buffer_obs[n:p],
        action=agent._buffer_action[n:p],
        next_obs=agent._buffer_next_obs[n:p],
    )
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(
            {
                "retrain_count": count,
                "date": str(date),
                "reward_mode": agent._reward_mode,
                "obs_dim": agent._obs_dim,
                "offline_transitions": n,
                "online_transitions_at_checkpoint": p - n,
                "policy_head": "ppo_explore (retrained)",
                "salvaged_from": os.path.abspath(args.checkpoint),
                "note": (
                    "Reconstructed from the agent state checkpoint after the "
                    "live archive was skipped. network/obs_norm/model ARE the "
                    "artifacts of retrain {count}, which is the most recent one "
                    "(nothing has overwritten them since). But the buffer is as "
                    "of the checkpoint, not the retrain: "
                    "online_transitions_at_checkpoint counts rows collected up "
                    "to {ckpt_time}, which may exceed what actually drove that "
                    "retrain. ppo_updates/model_update_steps are not recorded "
                    "in the checkpoint."
                ).format(
                    count=count,
                    ckpt_time=_mtime(args.checkpoint),
                ),
            },
            f,
            indent=1,
        )
    print(f"salvaged retrain {count} ({date}) -> {out}", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
