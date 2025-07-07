import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION, DIM_ACTION
from utils.functions import normalize


class TimeDLIPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dli = 0.0

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        dli = np.clip(normalize(self.dli, 0, 66), 0, 1)
        local_time = epoch_time.astimezone(self.tz)
        hours_since_start = (local_time.hour - 9) + (local_time.minute / 60)
        hours_normalized = hours_since_start / 11.0
        return np.array([hours_normalized, dli])

    async def step(self, action):
        if np.array_equal(action, BALANCED_ACTION):
            self.dli += 1.0
        elif np.array_equal(action, DIM_ACTION):
            self.dli += 0.5
        if self.get_local_time().hour == 9 and self.get_local_time().minute == 30:
            self.dli = 0.0

        return await super().step(action)
