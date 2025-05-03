import asyncio
import logging
from typing import Any

from RlGlue import RlGlue
from RlGlue.environment import BaseEnvironment
from RlGlue.rl_glue import Interaction

from utils.RlGlue.agent import BaseAgent, BaseAsyncAgent
from utils.RlGlue.environment import BaseAsyncEnvironment

logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)


background_tasks = set()


class AsyncRLGlue:
    def __init__(self, agent: BaseAsyncAgent, env: BaseAsyncEnvironment):
        self.environment = env
        self.agent = agent

        self.last_action: Any = None
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        self.total_steps: int = 0
        self.num_episodes: int = 0
        self.lock = asyncio.Lock()

    async def start(self):
        self.num_steps = 0
        self.total_reward = 0

        s, env_info = await self.environment.start()
        self.last_action, agent_info = await self.agent.start(s)
        plan_task = asyncio.create_task(self.plan())
        background_tasks.add(plan_task)
        info = {**env_info, **agent_info}

        return s, self.last_action, info

    async def step(self) -> Interaction:
        assert (
            self.last_action is not None
        ), "Action is None; make sure to call glue.start() before calling glue.step()."
        reward, s, term, env_info = await self.environment.step(self.last_action)

        self.total_reward += reward

        self.num_steps += 1
        self.total_steps += 1
        if term:
            return await self.end(reward, s, term, env_info)
        async with self.lock:
            self.last_action, agent_info = await self.agent.step(reward, s, env_info)
        info = {**env_info, **agent_info}
        return Interaction(
            o=s,
            a=self.last_action,
            t=term,
            r=reward,
            extra=info,
        )

    async def plan(self):
        try:
            while True:
                await asyncio.sleep(0)
                async with self.lock:
                    await self.agent.plan()
        except asyncio.CancelledError:
            pass

    async def end(self, reward: float, s: Any, term: bool, env_info: dict) -> Interaction:
        self.num_episodes += 1
        agent_info = await self.agent.end(reward, env_info)
        for task in background_tasks:
            task.cancel()
        info = {**env_info, **agent_info}
        return Interaction(
            o=s,
            a=None,
            t=term,
            r=reward,
            extra=info,
        )

    async def runEpisode(self, max_steps: int = 0):
        is_terminal = False

        await self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = await self.step()
            is_terminal = rl_step_result.t

        # even at episode cutoff, this still counts as completing an episode
        if not is_terminal:
            self.num_episodes += 1

        return is_terminal


class LoggingRlGlue(RlGlue):

    def __init__(self, agent: BaseAgent, env: BaseEnvironment):
        super().__init__(agent, env)
        self.agent = agent
        self.environment = env

    def start(self):
        self.num_steps = 0
        self.total_reward = 0

        s, env_info = self.environment.start()
        self.last_action, agent_info = self.agent.start(s)
        info = {**env_info, **agent_info}
        return s, self.last_action, info

    def step(self) -> Interaction:
        assert self.last_action is not None, 'Action is None; make sure to call glue.start() before calling glue.step().'
        (reward, s, term, env_info) = self.environment.step(self.last_action)

        self.total_reward += reward

        self.num_steps += 1
        self.total_steps += 1
        if term:
            self.num_episodes += 1
            self.agent.end(reward, env_info)
            return Interaction(
                o=s, a=None, t=term, r=reward, extra=env_info,
            )

        self.last_action, agent_info = self.agent.step(reward, s, env_info)
        info = {**env_info, **agent_info}
        return Interaction(
            o=s, a=self.last_action, t=term, r=reward, extra=info,
        )
