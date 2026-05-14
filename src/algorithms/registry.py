from typing import Type
from algorithms.RecipeAgent import RecipeAgent
from algorithms.Incubator import Incubator
from utils.RlGlue.agent import BaseAsyncAgent

def getAgent(name) -> Type[BaseAsyncAgent]:
    if name.startswith("Incubator"):
        return Incubator

    if name.startswith("Recipe"):
        return RecipeAgent

    raise Exception("Unknown algorithm")
