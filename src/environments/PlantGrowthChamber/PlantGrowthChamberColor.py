import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION


class PlantGrowthChamberColor(PlantGrowthChamber):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)

        return [local_time, df["mean_clean_area"]]

    async def step(self, action: float | np.ndarray):
        if isinstance(action, np.ndarray):
            return await super().step(action)

        # map scalar action in [-1, 1] to  the 1D color action space (fixed at 105 ppfd)
        if action == 0:
            color_action = BALANCED_ACTION_105
        elif action < 0:  # adds blue
            color_action = (1 - abs(action)) * BALANCED_ACTION_105 + abs(
                action
            ) * BLUE_ACTION
        elif action > 0:  # adds red
            color_action = (1 - action) * BALANCED_ACTION_105 + action * RED_ACTION

        return await super().step(color_action)
