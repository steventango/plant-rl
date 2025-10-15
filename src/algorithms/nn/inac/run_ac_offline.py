import argparse
import os
from pathlib import Path

import numpy as np

from algorithms.nn.inac.agent.in_sample import train
from algorithms.nn.inac.utils import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--env_name", default="Ant", type=str)
    parser.add_argument("--dataset", default="medexp", type=str)
    parser.add_argument("--discrete_control", default=True, type=bool)
    parser.add_argument("--state_dim", default=5, type=int)
    parser.add_argument("--action_dim", default=3, type=int)
    parser.add_argument("--tau", default=0.1, type=float)

    parser.add_argument("--max_steps", default=1000000, type=int)
    parser.add_argument("--log_interval", default=10000, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--hidden_units", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--use_target_network", default=1, type=int)
    parser.add_argument("--target_network_update_freq", default=1, type=int)
    parser.add_argument("--polyak", default=0.995, type=float)
    parser.add_argument("--evaluation_criteria", default="return", type=str)
    parser.add_argument("--info", default="0", type=str)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    cfg = parser.parse_args()

    np.random.seed(cfg.seed)

    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = "data/JAX/output/{}/{}/{}/{}_run".format(
        cfg.env_name, cfg.dataset, cfg.info, cfg.seed
    )
    cfg.exp_path = os.path.join(project_root, exp_path)
    os.makedirs(cfg.exp_path, exist_ok=True)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg, cfg.exp_path)
    logger.log_config(cfg)

    # Initializing the agent and running the experiment
    train(
        discrete_control=cfg.discrete_control,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_units=cfg.hidden_units,
        learning_rate=cfg.learning_rate,
        tau=cfg.tau,
        polyak=cfg.polyak,
        exp_path=Path(cfg.exp_path),
        seed=cfg.seed,
        env_fn=cfg.env_fn,
        timeout=cfg.timeout,
        gamma=cfg.gamma,
        offline_data=cfg.offline_data,
        batch_size=cfg.batch_size,
        use_target_network=cfg.use_target_network,
        target_network_update_freq=cfg.target_network_update_freq,
        logger=cfg.logger,
        max_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        weight_decay=cfg.weight_decay,
    )
