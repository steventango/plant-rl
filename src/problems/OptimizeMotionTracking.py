from algorithms.MotionTrackingControllerWrapper import (
    MotionTrackingControllerWrapper,
    MotionTrackingControllerWrapper_NoTracking
)
from algorithms.registry import getAgent
from problems.BaseAsyncProblem import BaseAsyncProblem
from utils.RlGlue.agent import BaseAsyncAgent
from experiment.ExperimentModel import ExperimentModel
from PyExpUtils.collection.Collector import Collector
from environments.PlantGrowthChamber.CVPlantGrowthChamberIntensity import (
    CVPlantGrowthChamberIntensity_MotionTracking as Env,
)

class OptimizeMotionTracking(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 1
        self.observations = (2,)

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
            if "should_track" not in self.env_params:
                agent = MotionTrackingControllerWrapper(agent)
            elif self.env_params["should_track"] == True:
                agent = MotionTrackingControllerWrapper(agent)
            elif self.env_params["should_track"] == False:
                agent = MotionTrackingControllerWrapper_NoTracking(agent)
            else:
                raise ValueError(
                    "Invalid input for should_track."
                )

        self.agent = agent
        return self.agent
