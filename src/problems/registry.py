from importlib import import_module

def getProblem(name):

    mod = import_module(f'problems.{name}')

    return getattr(mod, name)
