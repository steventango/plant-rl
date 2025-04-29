import numpy as np
import pandas as pd

import wandb


def expand(key, value):
    if isinstance(value, np.ndarray):
        result = {}
        for idx in np.ndindex(value.shape):
            idx_str = ".".join(map(str, idx))
            result[f"{key}.{idx_str}"] = value[idx]
        return result
    if isinstance(value, (list, tuple)):
        return {f"{key}.{i}": v for i, v in enumerate(value)}
    else:
        return {key: value}


def log(env, glue, wandb_run, s, a, info, r=None):
    expanded_info = {}
    for key, value in info.items():
        if isinstance(value, pd.DataFrame):
            table = wandb.Table(dataframe=value)
            expanded_info.update({key: table})
        elif isinstance(value, np.ndarray):
            if value.size < 16:
                expanded_info.update(expand(key, value))
        else:
            expanded_info.update(expand(key, value))
    data = {
        **expand("state", s),
        **expand("action", a),
        "steps": glue.num_steps,
        **expanded_info,
    }
    if hasattr(env, "time"):
        data["time"] = env.time
    if hasattr(env, "image"):
        data["raw_image"] = wandb.Image(env.image)
    if r is not None:
        data["reward"] = r
    wandb_run.log(data)
