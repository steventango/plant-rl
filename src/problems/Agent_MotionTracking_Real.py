from algorithms.motion_tracking.MotionTrackingWrapper import (
    MotionTrackingWrapper,
    MotionTrackingWrapper_NoTracking,
)
from algorithms.registry import getAgent
from problems.BaseAsyncProblem import BaseAsyncProblem
from utils.RlGlue.agent import BaseAsyncAgent
from experiment.ExperimentModel import ExperimentModel
from PyExpUtils.collection.Collector import Collector

from environments.PlantGrowthChamber.CVPlantGrowthChamberIntensity import (
    CVPlantGrowthChamberIntensity_MotionTracking as Env,
 )

class Agent_MotionTracking_Real(BaseAsyncProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(**self.env_params)
        self.actions = 3   # RL agent's action space = [decrease, keep, increase] intensity
        self.observations = (143,)  # RL agent's state space = whole day's area measurements

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
            if "should_track" not in self.params:
                agent = MotionTrackingWrapper((2,), 1, self.params, self.collector, self.seed, agent)
            elif self.params["should_track"]:
                agent = MotionTrackingWrapper((2,), 1, self.params, self.collector, self.seed, agent)
            elif not self.params["should_track"]:
                agent = MotionTrackingWrapper_NoTracking((2,), 1, self.params, self.collector, self.seed, agent)
            else:
                raise ValueError("Invalid input for should_track.")

        self.agent = agent
        return self.agent
