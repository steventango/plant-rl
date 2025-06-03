import asyncio
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from RlGlue import RlGlue
from RlGlue.environment import BaseEnvironment
from RlGlue.rl_glue import Interaction

from utils.logger import expand
from utils.RlGlue.agent import BaseAgent, BaseAsyncAgent
from utils.RlGlue.environment import BaseAsyncEnvironment

logger = logging.getLogger('rlglue')
logger.setLevel(logging.DEBUG)


background_tasks = set()
lock = asyncio.Lock()
default_save_keys = {"left", "right"}


class AsyncRLGlue:
    def __init__(self, agent: BaseAsyncAgent, env: BaseAsyncEnvironment, dataset_path: Path, images_save_keys: set[str] | None):
        self.environment = env
        self.agent = agent

        self.dataset_path = dataset_path
        self.is_mock_env = self.environment.__class__.__name__.startswith("Mock")
        if not self.is_mock_env:
            dataset_path.mkdir(parents=True, exist_ok=True)
        if images_save_keys is None:
            self.images_save_keys = default_save_keys
        else:
            self.images_save_keys = images_save_keys

        self.last_action: Any = None
        self.last_interaction: Interaction | None = None
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        self.total_steps: int = 0
        self.num_episodes: int = 0

    async def start(self):
        self.num_steps = 0
        self.total_reward = 0

        s, env_info = await self.environment.start()
        self.last_action, agent_info = await self.agent.start(s)
        plan_task = asyncio.create_task(self.plan())
        background_tasks.add(plan_task)
        info = {**env_info, **agent_info}
        self.last_interaction = Interaction(
            o=s,
            a=self.last_action,
            t=False,
            r=None,
            extra=info,
        )
        self.log()
        return self.last_interaction

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
        async with lock:
            self.last_action, agent_info = await self.agent.step(reward, s, env_info)
        info = {**env_info, **agent_info}
        self.last_interaction = Interaction(
            o=s,
            a=self.last_action,
            t=term,
            r=reward,
            extra=info,
        )
        self.log()
        return self.last_interaction

    async def plan(self):
        try:
            while True:
                await asyncio.sleep(0)
                async with lock:
                    await self.agent.plan()
        except asyncio.CancelledError:
            pass

    async def end(self, reward: float, s: Any, term: bool, env_info: dict) -> Interaction:
        self.num_episodes += 1
        agent_info = await self.agent.end(reward, env_info)
        for task in background_tasks:
            task.cancel()
        info = {**env_info, **agent_info}
        self.last_interaction = Interaction(
            o=s,
            a=None,
            t=term,
            r=reward,
            extra=info,
        )
        self.log()
        return self.last_interaction

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


    def append_csv(self, chk, raw_csv_path: Path, img_name: str, interaction: Interaction):
        data_dict = {
            "time": [self.environment.time],
            "frame": [self.num_steps],
            **expand("action", self.environment.last_action),
            "steps": [self.num_steps],
            "image_name": [img_name],
            "episode": [chk["episode"] if chk is not None else None],
        }
        expanded_info = {}
        if interaction is not None:
            interaction_data = {
                **expand("state", interaction.o),
                "agent_action": [interaction.a],
                "reward": [interaction.r],
                "terminal": [interaction.t],
                "return": [self.total_reward if interaction.t else None],
            }
            data_dict.update(interaction_data)
            for key, value in interaction.extra.items():
                if isinstance(value, pd.DataFrame):
                    continue
                elif isinstance(value, np.ndarray):
                    expanded_info.update(expand(key, value))
                else:
                    expanded_info.update(expand(key, value))
            data_dict.update(expanded_info)

        df = pd.DataFrame(
            data_dict
        )
        if interaction is not None:
            interaction.extra["df"].reset_index(inplace=True)
            interaction.extra["df"]["plant_id"] = interaction.extra["df"].index
            interaction.extra["df"]["frame"] = self.num_steps
            df = pd.merge(
                df,
                interaction.extra["df"],
                how="left",
                left_on=["frame"],
                right_on=["frame"],
            )
        if raw_csv_path.exists():
            df_old = pd.read_csv(raw_csv_path)
            shutil.copy(raw_csv_path, raw_csv_path.with_suffix(".bak"))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
                    category=FutureWarning,
                )
                df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(raw_csv_path, index=False)


    def save_images(self, dataset_path: Path, save_keys: set[str]):
        isoformat = self.environment.time.isoformat(timespec="seconds").replace(":", "")
        images_path = dataset_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)
        img_path = None
        for key, image in self.environment.images.items():
            if save_keys != "*" and key not in save_keys:
                continue
            img_path = images_path / f"{isoformat}_{key}.jpg"
            image = image.convert("RGB")
            image.save(img_path, "JPEG", quality=90)
        return img_path.name if img_path else None

    def log(self):
        # if self.is_mock_env:
        #     return

        img_name = self.save_images(self.dataset_path, self.images_save_keys)
        raw_csv_path = self.dataset_path / "raw.csv"
        self.append_csv(None, raw_csv_path, img_name, self.last_interaction)


class LoggingRlGlue(RlGlue):

    def __init__(self, agent: BaseAgent, env: BaseEnvironment):
        super().__init__(agent, env)
        self.agent = agent
        self.environment = env

    def start(self):
        self.num_steps = 0
        self.total_reward = 0

        s, env_info = self.environment.start()
        self.last_action, agent_info = self.agent.start(s, env_info)
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
            self.agent.end(reward, {**env_info})
            return Interaction(
                o=s, a=None, t=term, r=reward, extra=env_info,
            )

        self.last_action, agent_info = self.agent.step(reward, s, env_info)
        info = {**env_info, **agent_info}
        return Interaction(
            o=s, a=self.last_action, t=term, r=reward, extra=info,
        )
