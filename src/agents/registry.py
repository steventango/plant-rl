from typing import Type
from agents.RecipeAgent import RecipeAgent
from agents.Incubator import Incubator
from utils.RlGlue.agent import BaseAsyncAgent


def getAgent(name) -> Type[BaseAsyncAgent]:
    if name.startswith("Incubator"):
        return Incubator

    if name.startswith("Recipe"):
        return RecipeAgent

    raise Exception("Unknown algorithm")
