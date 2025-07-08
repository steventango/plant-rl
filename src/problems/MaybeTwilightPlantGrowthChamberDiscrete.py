from PyExpUtils.collection.Collector import Collector

from algorithms.PlantGrowthChamberAsyncAgentWrapper import (
    PlantGrowthChamberAsyncAgentWrapper,
    PlantGrowthChamberAsyncAgentWrapper_BrightTwilight,
    PlantGrowthChamberAsyncAgentWrapper_DimTwilight,
)
from algorithms.registry import getAgent
from environments.PlantGrowthChamber.CVPlantGrowthChamberDiscrete import (
    CVPlantGrowthChamberDiscrete as Env,
)
from experiment.ExperimentModel import ExperimentModel
from problems.BaseAsyncProblem import BaseAsyncProblem
from utils.RlGlue.agent import BaseAsyncAgent


class MaybeTwilightPlantGrowthChamberDiscrete(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 2
        self.observations = (13,)

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
            if "twilight_type" not in self.env_params:
                agent = PlantGrowthChamberAsyncAgentWrapper(agent)
            elif self.env_params["twilight_type"] == "bright":
                agent = PlantGrowthChamberAsyncAgentWrapper_BrightTwilight(agent)
            elif self.env_params["twilight_type"] == "dim":
                agent = PlantGrowthChamberAsyncAgentWrapper_DimTwilight(agent)
            else:
                raise ValueError("Invalid twilight_type.")

        self.agent = agent
        return self.agent
