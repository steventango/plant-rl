from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


def linear_interpolation(first, second, completed):
    return first + (second - first) * completed


class SpreadsheetAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.df = pd.read_excel(self.params["filepath"])
        self.df["datetime"] = self.df["Day"] * 86400 + self.df["Time"].apply(
            lambda x: x.hour * 3600 + x.minute * 60 + x.second
        )

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, observation: np.ndarray):
        current_time = observation[0]
        return self.get_action(current_time)

    def step(self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]):
        current_time = observation[0]
        return self.get_action(current_time)

    def end(self, reward: float, extra: Dict[str, Any]):
        pass

    def get_action(self, current_time: int):
        # current_day = current_time // 86400 % self.df["Day"].max()
        current_time = current_time % 86400

        if current_time >= self.df["datetime"].max():
            second_point = self.df.iloc[0]
            second_index = 0
        else:
            second_point = self.df[self.df["datetime"] > current_time].iloc[0]
            second_index = self.df.index[self.df["datetime"] > current_time][0]
        if second_index == 0:
            first_point = self.df.iloc[-1]
        else:
            first_point = self.df.iloc[second_index - 1]

        if first_point["datetime"] < second_point["datetime"]:
            region_completed = (current_time - first_point["datetime"]) / (
                second_point["datetime"] - first_point["datetime"]
            )
        else:
            region_completed = 1

        light_scaling_factor = linear_interpolation(first_point["Scaling"], second_point["Scaling"], region_completed)

        first_color = np.array(
            [
                first_point["Blue"],
                first_point["Cool_White"],
                first_point["Warm_White"],
                first_point["Orange_Red"],
                first_point["Red"],
                first_point["Far_Red"],
            ]
        )
        second_color = np.array(
            [
                second_point["Blue"],
                second_point["Cool_White"],
                second_point["Warm_White"],
                second_point["Orange_Red"],
                second_point["Red"],
                second_point["Far_Red"],
            ]
        )
        color = linear_interpolation(first_color, second_color, region_completed)

        action = color * light_scaling_factor
        return action
