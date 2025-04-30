import logging
import time

import numpy as np
from RlGlue import RlGlue
from RlGlue.environment import BaseEnvironment
from RlGlue.rl_glue import Interaction

from utils.RlGlue.agent import BasePlanningAgent
from utils.RlGlue.environment import BaseAsyncEnvironment

logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)

class PlanningRlGlue(RlGlue):
    def __init__(self, agent: BasePlanningAgent, env: BaseAsyncEnvironment, exp_params: dict):
        super().__init__(agent, BaseEnvironment())
        self.agent = agent
        self.environment = env
        self.exp_params = exp_params
        self.step_duration = exp_params.get('step_duration', 60)
        self.update_freq = exp_params.get('update_freq', 5)
        self.start_time = exp_params.get('start_time', None)
        self._total_steps = exp_params.get('total_steps', 0)

    def start(self):
        s, env_info = self.environment.start()
        self.last_action, agent_info = self.agent.start(s)
        info = {**env_info, **agent_info}

        if self.start_time is None:
            self.start_time = time.time()
        self.total_steps = self._total_steps
        return s, self.last_action, info

    def step(self) -> Interaction:
        assert (
            self.last_action is not None
        ), "Action is None; make sure to call glue.start() before calling glue.step()."
        if self.total_steps % self.update_freq == 0:
            self.environment.step_one(self.last_action)
        while time.time() < self.start_time + self.step_duration * (self.total_steps + 1):
            self.agent.plan()
        (reward, s, term, env_info) = self.environment.step_two()

        self.total_reward += reward

        self.num_steps += 1
        self.total_steps += 1
        if term:
            self.num_episodes += 1
            self.agent.end(reward, env_info)
            return Interaction(
                o=s,
                a=None,
                t=term,
                r=reward,
                extra=env_info,
            )

        agent_info = {}
        if self.total_steps % self.update_freq == 0:
            self.last_action, agent_info = self.agent.step(reward, s, env_info)
        info = {**env_info, **agent_info}
        return Interaction(
            o=s,
            a=self.last_action,
            t=term,
            r=reward,
            extra=info,
        )
