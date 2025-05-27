import os
import sys

sys.path.append(os.getcwd() + "/src")

import json
import lzma
import pickle
from pathlib import Path

import numpy as np

from utils.plotting import get_Q, plot_q, plot_q_diff

experiment_paths = Path(__file__).parent.glob("*.json")


for experiment_path in experiment_paths:
    checkpoint_path = (
        Path("checkpoints/results") / experiment_path.relative_to(Path("experiments")).with_suffix("") / "0/chk.pkl.xz"
    )
    if not checkpoint_path.exists():
        print(f"Checkpoint file {checkpoint_path} does not exist. Skipping {experiment_path}.")
        continue
    with open(experiment_path, "r") as f:
        config = json.load(f)
        representation_config = config["metaParameters"]["representation"]
    with lzma.open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    weights = checkpoint["a"].w

    tile_coder = checkpoint["a"].tile_coder

    daytime_observation_space = np.linspace(0, 1, 12 * 6, endpoint=True)
    area_observation_space = np.linspace(0, 1, 100, endpoint=True)

    num_actions = weights.shape[0]
    Q = get_Q(weights, tile_coder, daytime_observation_space, area_observation_space, num_actions)

    plot_q(daytime_observation_space, area_observation_space, Q)

    # Calculate the difference between Q[s, 1] and Q[s, 0]
    if num_actions >= 2:
        Q_diff = Q[:, :, 1] - Q[:, :, 0]
    else:
        print("Not enough actions to compute difference. Plotting Q[s,0] instead.")
        Q_diff = Q[:, :, 0]

    # plot Q value differences
    plot_q_diff(daytime_observation_space, area_observation_space, Q_diff)
