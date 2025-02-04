from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC
from algorithms.ContinuousRandomAgent import ContinuousRandomAgent
from algorithms.DiscreteRandomAgent import DiscreteRandomAgent
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC


def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQN"):
        return DQN

    if name == 'EQRC':
        return EQRC

    if name == 'ESARSA':
        return ESARSA

    if name == 'SoftmaxAC':
        return SoftmaxAC

    if name == 'DiscreteRandom':
        return DiscreteRandomAgent

    if name == 'ContinuousRandom':
        return ContinuousRandomAgent

    raise Exception('Unknown algorithm')
