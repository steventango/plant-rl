import numpy as np  # type: ignore

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION


class PlantGrowthChamberColor(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_map = {
            0: BALANCED_ACTION_105,
            1: BLUE_ACTION,
            2: RED_ACTION,
        }

    async def step(self, action: int | np.ndarray):
        if isinstance(action, np.ndarray):
            return await super().step(action)

        return await super().step(self.action_map[action])  # type: ignore
