from datetime import datetime
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


def linear_interpolation(first, second, completed):
    return first + (second - first) * completed


class SpreadsheetAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.df = pd.read_excel(self.params["filepath"])
        self.compatibility_mode = self.params.get("compatibility_mode", False)
        self.df["datetime"] = self.df["Day"] * 86400 + pd.to_datetime(
            self.df["Time"], format="%H:%M:%S"
        ).apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        tz = ZoneInfo(self.params.get("timezone", "Etc/UTC"))
        dt = datetime.now(tz)
        offset = dt.utcoffset()
        assert offset is not None
        self.offset = offset.total_seconds()

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(
        self, observation: np.ndarray, extra: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        utc_time = observation[0]
        return self.get_action(utc_time), {}

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        utc_time = observation[0]
        return self.get_action(utc_time), {}

    def end(self, reward: float, extra: Dict[str, Any]):
        return {}

    def get_action(self, utc_time: float):
        local_time = utc_time + self.offset
        cycle_length = 86400 * (self.df["Day"].max() + 1)
        clock_time = local_time % cycle_length

        if clock_time >= self.df["datetime"].max():
            second_point = self.df.iloc[0]
            first_point = self.df.iloc[-1]
        else:
            second_point = self.df[self.df["datetime"] > clock_time].iloc[0]
            second_index = self.df[self.df["datetime"] > clock_time].index[0]
            first_point = self.df.iloc[second_index - 1]

        first_datetime = first_point["datetime"]
        second_datetime = second_point["datetime"]

        if second_datetime < first_datetime:
            second_datetime += cycle_length

        if clock_time < first_datetime:
            clock_time += cycle_length

        region_completed = (clock_time - first_datetime) / (
            second_datetime - first_datetime
        )

        light_scaling_factor = linear_interpolation(
            first_point["Scaling"], second_point["Scaling"], region_completed
        )

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

        if self.compatibility_mode:
            action *= 2

        return action
