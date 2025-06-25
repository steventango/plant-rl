'''
This implementation is taken from the official codebase for Greedy AC
which can be found at https://github.com/samuelfneumann/GreedyAC/tree/master
'''
from typing import Any, Dict, Tuple

import numpy as np

from algorithms.BaseAgent import BaseAgent
from PyExpUtils.collection.Collector import Collector
from .agent.GreedyACDiscrete import GreedyACDiscrete


class GreedyAC(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        assert len(observations) == 1, "GreedyAC currently only supports flat observations"
        self.state = None
        self.action = None
        self.greedy_ac = GreedyACDiscrete(
            num_inputs = observations[0],
            num_actions = actions,
            gamma = self.gamma,
            tau = params['tau'],
            policy = params['policy_type'],
            target_update_interval = params['target_update_interval'],
            critic_lr = params['critic_lr'],
            actor_lr_scale = params['actor_lr_scale'],
            actor_hidden_dim = params['hidden_dim'],
            critic_hidden_dim = params['hidden_dim'],
            actor_n_hidden= params['n_hidden'],
            critic_n_hidden= params['n_hidden'],
            replay_capacity = params['replay_capacity'],
            seed = seed,
            batch_size = params['batch_size'],
            beta1 = params['beta1'],
            beta2 = params['beta2'],
            cuda=False,
            clip_stddev=1000,
            init=None,
            entropy_from_single_sample=True,
            activation="relu")

    def start(self, observation: Any, extra: Dict[str, Any]) -> int:
        self.state = observation
        self.action = self.greedy_ac.sample_action(self.state)
        return self.action

    def step(self, reward: float, observation: Any, extra: Dict[str, Any]) -> int:
        self.greedy_ac.update(
            state=self.state,
            action=self.action,
            reward = reward,
            next_state=observation,
            done_mask=True
            )
        self.state = observation
        self.action = self.greedy_ac.sample_action(self.state)

        return self.action

    def plan(self):
        super().plan()
        self.greedy_ac.plan()

    def end(self, reward: float, extra: Dict[str, Any]):
        self.greedy_ac.update(
            state=self.state,
            action=self.action,
            reward = reward,
            next_state=np.zeros(self.observations),
            done_mask=False
            )
        self.state = None
        self.action = None
        return {}
