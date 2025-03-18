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
        result = super().start()
        if self.start_time is None:
            self.start_time = time.time()
        self.total_steps = self._total_steps
        s, last_action = result
        logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] env start: state[0] {int(s[0])}")
        logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] env start: action {last_action}")
        return result

    def step(self) -> Interaction:
        assert (
            self.last_action is not None
        ), "Action is None; make sure to call glue.start() before calling glue.step()."
        if self.total_steps % self.update_freq == 0:
            logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] env step one start: action {self.last_action}")
            self.environment.step_one(self.last_action)
            logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] env step one end")
        logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] agent planning start")
        while time.time() < self.start_time + self.step_duration * (self.total_steps + 1):
            self.agent.plan()
        logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] agent planning end")
        logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] env step two start")
        (reward, s, term, extra) = self.environment.step_two()
        logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] env step two end")
        logger.debug(f"state[0] {int(s[0])}")
        logger.debug(f"reward {reward}")

        self.total_reward += reward

        self.num_steps += 1
        self.total_steps += 1
        if term:
            self.num_episodes += 1
            self.agent.end(reward, extra)
            return Interaction(
                o=s,
                a=None,
                t=term,
                r=reward,
                extra=extra,
            )

        if self.total_steps % self.update_freq == 0:
            logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] agent step start")
            self.last_action = self.agent.step(reward, s, extra)
            logger.debug(f"#{self.total_steps} [{time.time() - self.start_time:.2f} s] agent step end")
        return Interaction(
            o=s,
            a=self.last_action,
            t=term,
            r=reward,
            extra=extra,
        )
