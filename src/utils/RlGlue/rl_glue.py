import logging
import time

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

    def start(self):
        result = super().start()
        self.start_time = time.time()
        return result

    def step(self) -> Interaction:
        assert (
            self.last_action is not None
        ), "Action is None; make sure to call glue.start() before calling glue.step()."
        if self.num_steps % self.update_freq == 0:
            logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] env step one start: action {self.last_action}")
            self.environment.step_one(self.last_action)
            logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] env step one end")
        logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] agent planning start")
        while time.time() < self.start_time + self.step_duration * (self.num_steps + 1):
            self.agent.plan()
        logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] agent planning end")
        if hasattr(self.agent, 'greedy_ac'):
            logger.debug(f"q loss: {self.agent.greedy_ac.q_loss}")
            logger.debug(f"policy loss: {self.agent.greedy_ac.policy_loss}")
        logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] env step two start")
        (reward, s, term, extra) = self.environment.step_two()
        logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] env step two end: state[0] {s[0]}, reward {reward}")

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

        if self.num_steps % self.update_freq == 0:
            logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] agent step start")
            self.last_action = self.agent.step(reward, s, extra)
            logger.debug(f"#{self.num_steps} [{time.time() - self.start_time} s] agent step end")
        return Interaction(
            o=s,
            a=self.last_action,
            t=term,
            r=reward,
            extra=extra,
        )
