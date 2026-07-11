"""Training hyperparameters for nightly in-agent retraining.

These reproduce the offline runs that produced the deployed E20/P1 checkpoints
(model-uncertainty-exploration `main.py --offline --dataset plant-data/visu-v2-v27
--ppo.gamma 0.8 model:enn --model.use_layer_norm`, verified against the runs'
logged hparams). Per-zone overrides come from the experiment JSON's "mbrl" dict.
"""

PPO_DEFAULTS: dict = {
    "LR": 3e-4,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 1e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.8,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "HIDDEN_DIM": 64,
    "ACTIVATION": "tanh",
    "USE_LAYER_NORM": False,
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": True,
}

ENN_DEFAULTS: dict = {
    "LR": 1e-3,
    "HIDDEN_DIM": 64,
    "LEARNABLE_HIDDEN_DIM": 15,
    "PRIOR_HIDDEN_DIM": 5,
    "INDEX_DIM": 8,
    "UPDATE_STEPS": 10000,
    "USE_LAYER_NORM": True,
    "MINIBATCH_SIZE": 256,
}

MODEL_ENV_DEFAULTS: dict = {
    "PREDICTION_MODE": "sample",
    "EXPLORE_BONUS": "eig",
    "RESET_SOURCE": "init",
    "MAX_STEPS_IN_EPISODE": 14,
    # Explore-policy reward weights: alpha * extrinsic + beta * intrinsic.
    "ALPHA": 0.0,
    "BETA": 1.0,
}


def merged(defaults: dict, overrides: dict | None) -> dict:
    """Merge JSON overrides (case-insensitive keys) over a defaults dict."""
    out = dict(defaults)
    for key, value in (overrides or {}).items():
        out[key.upper()] = value
    return out
