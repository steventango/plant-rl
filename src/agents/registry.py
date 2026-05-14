from typing import Type
from agents.ScheduleAgent import ScheduleAgent
from agents.Incubator import Incubator
from utils.RlGlue.agent import BaseAsyncAgent


def getAgent(name) -> Type[BaseAsyncAgent]:
    if name.startswith("Incubator"):
        return Incubator

    if name.startswith("Schedule"):
        return ScheduleAgent

    raise Exception("Unknown algorithm")
