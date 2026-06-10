from algorithms.PlantGrowthChamberAsyncAgentWrapper import (
    PlantGrowthChamberAsyncAgentWrapper,
    PlantGrowthChamberAsyncAgentWrapper_BrightTwilight,
    PlantGrowthChamberAsyncAgentWrapper_DimTwilight,
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
            twilight_type = self.env_params.get("twilight_type")
            if twilight_type is None:
                agent = PlantGrowthChamberAsyncAgentWrapper(agent, self.env)
            elif twilight_type == "bright":
                agent = PlantGrowthChamberAsyncAgentWrapper_BrightTwilight(
                    agent, self.env
                )
            elif twilight_type == "dim":
                agent = PlantGrowthChamberAsyncAgentWrapper_DimTwilight(
                    agent, self.env
                )
            else:
                raise ValueError(f"Invalid twilight_type: {twilight_type}")
        self.agent = agent
        return self.agent
