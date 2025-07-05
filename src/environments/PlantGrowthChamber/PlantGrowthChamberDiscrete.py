import numpy as np  # type: ignore

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION, DIM_ACTION


class PlantGrowthChamberDiscrete(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_map = {
            0: DIM_ACTION,
            1: BALANCED_ACTION,
        }

    async def step(self, action: int | np.ndarray):
        if isinstance(action, np.ndarray):
            return await super().step(action)
        action = self.action_map[action]
        return await super().step(action)  # type: ignore
