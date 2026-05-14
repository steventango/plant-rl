from datetime import timedelta

import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION_105


class PlantGrowthChamberIntensity(PlantGrowthChamber):
    def __init__(self, *args, step_duration_minutes: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_spectrum = BALANCED_ACTION_105
        self.duration = timedelta(minutes=step_duration_minutes)

    async def step(self, action: float | np.ndarray):
        if isinstance(action, np.ndarray) and action.ndim > 0:
            return await super().step(action)
        action = self.reference_spectrum * action
        return await super().step(action)
