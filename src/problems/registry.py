from importlib import import_module
from problems.BaseProblem import BaseProblem

def getProblem(name: str) -> type[BaseProblem]:
    
    mod = import_module(f'problems.{name}')

    return getattr(mod, name)
