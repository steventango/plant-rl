import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION


class PlantGrowthChamberColorTriangle(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis = np.column_stack([RED_ACTION, BALANCED_ACTION_105, BLUE_ACTION])

    async def step(self, action: np.ndarray):
        if action.shape[0] == 6:
            return await super().step(action)
        action = self.basis @ action
        return await super().step(action)
