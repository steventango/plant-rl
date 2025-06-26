from algorithms.PlantGrowthChamberAsyncAgentWrapper import (
    PlantGrowthChamberAsyncAgentWrapper,
)
from algorithms.registry import getAgent
from problems.BaseAsyncProblem import BaseAsyncProblem
from utils.RlGlue.agent import BaseAsyncAgent


class BasePlantGrowthChamberAsyncProblem(BaseAsyncProblem):
    def getAgent(self):
        if self.agent is not None:
            return self.agent

        if self.gamma is not None:
            self.params["gamma"] = self.gamma

        Agent = getAgent(self.exp.agent)
        agent = Agent(
            self.observations, self.actions, self.params, self.collector, self.seed
        )
        if not isinstance(agent, BaseAsyncAgent):
            agent = PlantGrowthChamberAsyncAgentWrapper(agent)
        self.agent = agent
        return self.agent
