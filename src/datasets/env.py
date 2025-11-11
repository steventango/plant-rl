from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces
from gymnasium.core import ObsType


class MockEnv(gym.Env):
    def __init__(
        self,
        df: pl.DataFrame,
        include_action_traces: bool = True,
        use_continuous_actions: bool = False,
    ):
        super().__init__()
        self.df = df.sort("experiment", "zone", "plant_id", "time")
        self.episode_keys = (
            df.select(["experiment", "zone", "plant_id"])
            .unique()
            .sort(["experiment", "zone", "plant_id"])
            .rows()
        )
        self.current_episode_index = 0
        self.current_episode_key = None
        self.current_row_index = 0
        self.plant_df = None
        self.was_truncated = False
        self.truncated_episode_key = None
        self.truncated_row_index = 0
        self.completed_episodes = set()  # Track completed episodes
        self.include_action_traces = include_action_traces
        self.use_continuous_actions = use_continuous_actions
        # Compute global min and max for clean_area normalization
        clean_area_min = df["clean_area"].min()
        clean_area_max = df["clean_area"].max()
        print(f"Global clean_area min: {clean_area_min}, max: {clean_area_max}")
        self.clean_area_min = (
            float(clean_area_min) if clean_area_min is not None else 0.0
        )  # type: ignore
        self.clean_area_max = (
            float(clean_area_max) if clean_area_max is not None else 1.0
        )  # type: ignore
        # Compute global min and max for day normalization
        day_min = df["day"].min()
        day_max = df["day"].max()
        self.day_min = float(day_min) if day_min is not None else 0.0  # type: ignore
        self.day_max = float(day_max) if day_max is not None else 1.0  # type: ignore
        # Set observation space based on include_action_traces
        obs_dim = 4  # clean_area, 3 for one-hot action
        if self.include_action_traces:
            obs_dim += 3  # 1 trace (0.5) * 3 values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        if use_continuous_actions:
            # Action space: [red_coef, white_coef, blue_coef]
            self.action_space = spaces.Box(
                low=0, high=1, shape=(3,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(3)

    def _get_observation(self) -> Any:
        # return a vector with the following values:
        # clean_area
        # action (one-hot encoded for discrete, continuous coefficients for continuous, from previous row)
        # action_trace_0.5 (one-hot encoded for discrete, continuous for continuous, smoothed) - if include_action_traces
        if self.plant_df is None or self.current_row_index >= self.plant_df.height:
            obs_dim = 4 if not self.include_action_traces else 7
            return np.zeros((obs_dim,), dtype=np.float32)

        row = self.plant_df.slice(self.current_row_index, 1)
        if row.is_empty():
            obs_dim = 4 if not self.include_action_traces else 7
            return np.zeros((obs_dim,), dtype=np.float32)

        clean_area = row["clean_area"][0] if row["clean_area"][0] is not None else 0.0
        # Normalize clean_area to [0, 1]
        if self.clean_area_max > self.clean_area_min:
            clean_area = (clean_area - self.clean_area_min) / (
                self.clean_area_max - self.clean_area_min
            )
        else:
            clean_area = 0.0  # If min == max, set to 0

        # Get action from the previous row
        if self.use_continuous_actions:
            # Use continuous action coefficients
            action_values = [0.0, 0.0, 0.0]
            action_trace_05 = [0.0, 0.0, 0.0]

            if self.current_row_index > 0:
                prev_row = self.plant_df.slice(self.current_row_index - 1, 1)
                if not prev_row.is_empty():
                    action_values = [
                        prev_row["red_coef"][0]
                        if prev_row["red_coef"][0] is not None
                        else 0.0,
                        prev_row["white_coef"][0]
                        if prev_row["white_coef"][0] is not None
                        else 0.0,
                        prev_row["blue_coef"][0]
                        if prev_row["blue_coef"][0] is not None
                        else 0.0,
                    ]
                    if self.include_action_traces:
                        action_trace_05 = [
                            prev_row["red_coef_trace_0.5"][0]
                            if prev_row["red_coef_trace_0.5"][0] is not None
                            else 0.0,
                            prev_row["white_coef_trace_0.5"][0]
                            if prev_row["white_coef_trace_0.5"][0] is not None
                            else 0.0,
                            prev_row["blue_coef_trace_0.5"][0]
                            if prev_row["blue_coef_trace_0.5"][0] is not None
                            else 0.0,
                        ]

            obs_list = [[clean_area], action_values]
            if self.include_action_traces:
                obs_list.append(action_trace_05)
        else:
            # Use discrete action (one-hot encoded)
            discrete_action = None
            discrete_action_trace_05 = [0.0, 0.0, 0.0]

            if self.current_row_index > 0:
                prev_row = self.plant_df.slice(self.current_row_index - 1, 1)
                if not prev_row.is_empty():
                    discrete_action = prev_row["discrete_action"][0]
                    if self.include_action_traces:
                        discrete_action_trace_05 = [
                            prev_row["discrete_action_trace_0_0.5"][0]
                            if prev_row["discrete_action_trace_0_0.5"][0] is not None
                            else 0.0,
                            prev_row["discrete_action_trace_1_0.5"][0]
                            if prev_row["discrete_action_trace_1_0.5"][0] is not None
                            else 0.0,
                            prev_row["discrete_action_trace_2_0.5"][0]
                            if prev_row["discrete_action_trace_2_0.5"][0] is not None
                            else 0.0,
                        ]

            discrete_action_one_hot = np.zeros((3,), dtype=np.float32)
            if discrete_action is not None and 0 <= discrete_action < 3:
                discrete_action_one_hot[int(discrete_action)] = 1.0

            obs_list = [
                [clean_area],
                discrete_action_one_hot,
            ]
            if self.include_action_traces:
                obs_list.append(discrete_action_trace_05)

        obs = np.concatenate(obs_list).astype(np.float32)
        return obs

    def _get_action(self) -> int | np.ndarray:
        if self.plant_df is None or self.current_row_index >= self.plant_df.height:
            if self.use_continuous_actions:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                return 0

        row = self.plant_df.slice(self.current_row_index, 1)
        if row.is_empty():
            if self.use_continuous_actions:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                return 0

        if self.use_continuous_actions:
            # Return continuous action coefficients
            red_coef = row["red_coef"][0] if row["red_coef"][0] is not None else 0.0
            white_coef = (
                row["white_coef"][0] if row["white_coef"][0] is not None else 0.0
            )
            blue_coef = row["blue_coef"][0] if row["blue_coef"][0] is not None else 0.0
            return np.array([red_coef, white_coef, blue_coef], dtype=np.float32)
        else:
            # Return discrete action
            discrete_action = row["discrete_action"][0]
            return int(discrete_action) if discrete_action is not None else 0

    def is_done(self) -> bool:
        """Check if all episodes have been completed"""
        return len(self.completed_episodes) >= len(self.episode_keys)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        super().reset(seed=seed)

        # If we were truncated, continue from where we left off
        if self.was_truncated and self.truncated_episode_key is not None:
            self.current_episode_key = self.truncated_episode_key
            self.current_row_index = self.truncated_row_index
            self.was_truncated = False
            self.truncated_episode_key = None
            self.truncated_row_index = 0

            # Get the plant data (should already be set, but refresh to be safe)
            self.plant_df = self.df.filter(
                (pl.col("experiment") == self.current_episode_key[0])
                & (pl.col("zone") == self.current_episode_key[1])
                & (pl.col("plant_id") == self.current_episode_key[2])
            ).sort("time")
        else:
            # Select the next episode (cycle through all unique experiment-zone-plant combinations)
            # Skip episodes that are already completed
            while self.current_episode_index < len(self.episode_keys):
                candidate_key = self.episode_keys[self.current_episode_index]
                self.current_episode_index += 1

                if candidate_key not in self.completed_episodes:
                    self.current_episode_key = candidate_key
                    break
            else:
                # All episodes completed, return None to indicate done
                return None, {"done": True}

            # Get all rows for this episode
            self.plant_df = self.df.filter(
                (pl.col("experiment") == self.current_episode_key[0])
                & (pl.col("zone") == self.current_episode_key[1])
                & (pl.col("plant_id") == self.current_episode_key[2])
            ).sort("time")
            self.current_row_index = 0

        obs = self._get_observation()
        info = {"action": self._get_action()}
        return obs, info

    def step(self, action: int | np.ndarray) -> Tuple[Any, float, bool, bool, dict]:
        # Get current row for reward and terminal flag
        row = self.plant_df.slice(self.current_row_index, 1)

        reward = float(row["reward"][0]) if row["reward"][0] is not None else 0.0
        terminal = bool(row["terminal"][0]) if row["terminal"][0] is not None else False
        truncated = (
            bool(row["truncated"][0]) if row["truncated"][0] is not None else False
        )

        # Move to next row
        self.current_row_index += 1

        # Check if we've reached the end of this plant's data
        if self.current_row_index >= self.plant_df.height:
            terminal = True

        # If truncated, save state to continue from this point in next reset
        if truncated and not terminal:
            self.was_truncated = True
            self.truncated_episode_key = self.current_episode_key
            self.truncated_row_index = self.current_row_index
        elif terminal and not truncated:
            # Episode completed naturally, mark it as done
            if self.current_episode_key is not None:
                self.completed_episodes.add(self.current_episode_key)

        obs = self._get_observation()
        info = {"action": self._get_action()}

        return obs, reward, terminal, truncated, info
