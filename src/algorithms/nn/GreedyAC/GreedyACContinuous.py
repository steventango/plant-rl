"""# type: ignore
This implementation is taken from the official codebase for Greedy AC
which can be found at https://github.com/samuelfneumann/GreedyAC/tree/master
"""

from typing import Any, Dict, Tuple

import numpy as np
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent

from .agent.GreedyACContinuous import GreedyACContinuous


class GreedyAC(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        assert len(observations) == 1, (
            "GreedyAC currently only supports flat observations"
        )
        self.state = None
        self.action = None
        self.greedy_ac = GreedyACContinuous(
            input_dim=observations[0],
            action_dim=actions,
            gamma=self.gamma,
            tau=params.get("tau", 0.01),
            entropy_scale=params.get("entropy_scale", 0.01),
            policy=params.get("policy_type", "dirichlet"),
            target_update_interval=params.get("target_update_interval", 1),
            critic_lr=params.get("critic_lr", 0.001),
            actor_lr_scale=params.get("actor_lr_scale", 1.0),
            actor_hidden_dim=params.get("hidden_dim", 64),
            critic_hidden_dim=params.get("hidden_dim", 64),
            actor_n_hidden=params.get("n_hidden", 2),
            critic_n_hidden=params.get("n_hidden", 2),
            replay_capacity=params.get("replay_capacity", 100000),
            seed=seed,
            batch_size=params.get("batch_size", 32),
            rho=params.get("rho", 0.1),
            num_samples=params.get("num_samples", 30),
            beta1=params.get("beta1", 0.9),
            beta2=params.get("beta2", 0.999),
            cuda=params.get("cuda", False),
            clip_stddev=params.get("clip_stddev", 1000),
            init=params.get("weight_init", "xavier_uniform"),
            entropy_from_single_sample=True,
            activation=params.get("activation", "relu"),
        )
        self.deterministic = params.get("deterministic", False)

    def start(self, observation: Any, extra: Dict[str, Any]) -> int:  # type: ignore
        self.state = observation
        self.action = self.get_action()
        return self.action, {}

    def step(self, reward: float, observation: Any, extra: Dict[str, Any]) -> int:  # type: ignore
        self.greedy_ac.update(
            state=self.state,
            action=self.action,
            reward=reward,
            next_state=observation,
            done_mask=True,
        )
        self.state = observation
        self.action = self.get_action()
        return self.action, {}

    def get_action(self):
        if self.deterministic:
            self.greedy_ac.eval()
            a = self.greedy_ac.sample_action(self.state)
            self.greedy_ac.train()
            return a
        else:
            return self.greedy_ac.sample_action(self.state)

    def plan(self):
        super().plan()
        self.greedy_ac.plan()

    def end(self, reward: float, extra: Dict[str, Any]):
        self.greedy_ac.update(
            state=self.state,
            action=self.action,
            reward=reward,
            next_state=np.zeros(self.observations),
            done_mask=False,
        )
        self.state = None
        self.action = None
        return {}
