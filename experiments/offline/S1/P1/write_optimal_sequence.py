import json
from pathlib import Path

import numpy as np

path = Path(__file__).parent / "optimal-sequence.json"

with open(path) as f:
    config = json.load(f)

total_days = 21
optimal_action = np.tile(np.hstack([np.ones(3 * 6), 2 * np.ones(6 * 6), np.ones(3 * 6)]), total_days)[:-1]
config["metaParameters"]["actions"] = json.dumps(optimal_action.tolist())

with open(path, "w") as f:
    json.dump(config, f, indent=4)
