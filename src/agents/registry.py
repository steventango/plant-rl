from typing import Type
from agents.SequenceAgent import SequenceAgent
from agents.Incubator import Incubator
from utils.RlGlue.agent import BaseAsyncAgent


def getAgent(name) -> Type[BaseAsyncAgent]:
    if name.startswith("Incubator"):
        return Incubator

    if name.startswith("Sequence"):
        return SequenceAgent

    raise Exception("Unknown algorithm")
