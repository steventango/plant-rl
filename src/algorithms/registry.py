from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC
from algorithms.ConstantAgent import ConstantAgent
from algorithms.ContinuousRandomAgent import ContinuousRandomAgent
from algorithms.DiscreteRandomAgent import DiscreteRandomAgent
from algorithms.SpreadsheetAgent import SpreadsheetAgent
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC
from algorithms.linear.ESARSA import ESARSA as LinearESARSA
from algorithms.linear.LinearQL import LinearQL#, LinearBatchQL
from algorithms.nn.GreedyAC.GreedyAC import GreedyAC

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQN"):
        return DQN

    if name == 'EQRC':
        return EQRC

    if name == 'ESARSA':
        return ESARSA

    if name == 'LinearESARSA':
        return LinearESARSA

    if name == 'LinearQL':
        return LinearQL
    '''
    if name == 'LinearBatchQL':
        return LinearBatchQL
    '''
    if name == 'SoftmaxAC':
        return SoftmaxAC

    if name == 'Constant':
        return ConstantAgent

    if name == 'DiscreteRandom':
        return DiscreteRandomAgent

    if name == 'ContinuousRandom':
        return ContinuousRandomAgent

    if name == 'Spreadsheet':
        return SpreadsheetAgent

    if name.startswith("GreedyAC"):
        return GreedyAC

    raise Exception('Unknown algorithm')
