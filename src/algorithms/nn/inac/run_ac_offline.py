import argparse
import os
from pathlib import Path

import minari
import numpy as np

from algorithms.nn.inac.agent.in_sample import train
from algorithms.nn.inac.utils import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default="plant-rl/continuous-v8", type=str)
    parser.add_argument("--discrete_control", default=False, type=bool)
    parser.add_argument(
        "--policy_type",
        default="dirichlet",
        type=str,
        choices=["normal", "dirichlet", "mixture_dirichlet"],
    )
    parser.add_argument("--state_dim", default=7, type=int)
    parser.add_argument("--action_dim", default=3, type=int)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--max_steps", default=100_000, type=int)
    parser.add_argument("--log_interval", default=10000, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--hidden_units", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--use_target_network", default=1, type=int)
    parser.add_argument("--target_network_update_freq", default=1, type=int)
    parser.add_argument("--polyak", default=0.995, type=float)
    parser.add_argument("--evaluation_criteria", default="return", type=str)
    parser.add_argument("--info", default="0", type=str)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--clip_grad_norm", default=1.0, type=float)
    cfg = parser.parse_args()

    np.random.seed(cfg.seed)

    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = f"data/JAX/output/{cfg.dataset}/{cfg.info}/{cfg.seed}_run"
    cfg.exp_path = os.path.join(project_root, exp_path)
    os.makedirs(cfg.exp_path, exist_ok=True)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg, cfg.exp_path)
    logger.log_config(cfg)

    # Load offline data
    offline_data = minari.load_dataset(cfg.dataset)

    # Initializing the agent and running the experiment
    train(
        discrete_control=cfg.discrete_control,
        policy_type=cfg.policy_type,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_units=cfg.hidden_units,
        learning_rate=cfg.learning_rate,
        tau=cfg.tau,
        polyak=cfg.polyak,
        exp_path=Path(cfg.exp_path),
        seed=cfg.seed,
        gamma=cfg.gamma,
        offline_data=offline_data,
        batch_size=cfg.batch_size,
        use_target_network=cfg.use_target_network,
        target_network_update_freq=cfg.target_network_update_freq,
        logger=cfg.logger,
        max_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm,
    )
