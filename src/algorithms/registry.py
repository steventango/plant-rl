from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.DynamicBatchDQN import DynamicBatchDQN
from algorithms.linear.linearDQN.LinearDynamicBatchDQN import LinearDynamicBatchDQN
from algorithms.nn.EQRC import EQRC
from algorithms.ConstantAgent import ConstantAgent
from algorithms.ContinuousRandomAgent import ContinuousRandomAgent
from algorithms.DiscreteRandomAgent import DiscreteRandomAgent
from algorithms.SequenceAgent import SequenceAgent
from algorithms.SpreadsheetAgent import SpreadsheetAgent
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.QL import QL
from algorithms.tc.tc_replay.ESARSA import ESARSA as ReplayESARSA
from algorithms.tc.tc_replay.QL import QL as QLReplay
from algorithms.tc.tc_offline.ESARSA import ESARSA as OfflineESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC
from algorithms.linear.ESARSA import ESARSA as LinearESARSA
from algorithms.linear.LinearQL import LinearQL
from algorithms.nn.GreedyAC.GreedyAC import GreedyAC

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQN"):
        return DQN

    if name == 'DynamicBatchDQN':
        return DynamicBatchDQN

    if name == 'LinearDynamicBatchDQN':
        return LinearDynamicBatchDQN

    if name == 'EQRC':
        return EQRC

    if name.startswith("ESARSA"):
        return ESARSA

    if name == 'QL':
        return QL

    if name == 'ReplayESARSA':
        return ReplayESARSA

    if name == 'QLReplay':
        return QLReplay

    if name.startswith("LinearESARSA"):
        return LinearESARSA

    if name.startswith("OfflineESARSA"):
        return OfflineESARSA

    if name == 'LinearQL':
        return LinearQL
    '''
    if name == 'LinearBatchQL':
        return LinearBatchQL
    '''
    if name == 'SoftmaxAC':
        return SoftmaxAC

    if name.startswith("Constant"):
        return ConstantAgent

    if name == 'DiscreteRandom':
        return DiscreteRandomAgent

    if name == 'ContinuousRandom':
        return ContinuousRandomAgent

    if name.startswith("Sequence"):
        return SequenceAgent

    if name.startswith("Spreadsheet"):
        return SpreadsheetAgent

    if name.startswith("GreedyAC"):
        return GreedyAC

    raise Exception('Unknown algorithm')
