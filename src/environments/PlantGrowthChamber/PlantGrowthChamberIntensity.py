from datetime import timedelta
import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.constants import BALANCED_ACTION_100


class PlantGrowthChamberIntensity(PlantGrowthChamber):
    def __init__(self, min_ppfd: float = 0.0, max_ppfd: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.min_ppfd = min_ppfd
        self.max_ppfd = max_ppfd

    def get_mean_area(self, plant_areas):
        if np.sum(self.last_action) == 0:
            return 0.0
        else:
            areas = np.array(plant_areas)
            low, high = np.percentile(areas, 25), np.percentile(areas, 75)
            trimmed = areas[(areas >= low) & (areas <= high)]
            return float(np.mean(trimmed)) if len(trimmed) > 0 else 0.0

    async def get_observation(self):  # type: ignore
        epoch_time, _, df = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)

        if not df.empty:
            plant_areas = df["area"].to_numpy().flatten()
            mean_area = self.get_mean_area(plant_areas)
        else:
            mean_area = 0.0

        return [local_time, mean_area]

    async def step(self, action: float | np.ndarray):
        if isinstance(action, np.ndarray):
            return await super().step(action)

        ppfd = self.min_ppfd + (action + 1) / 2 * (
            self.max_ppfd - self.min_ppfd
        )  # map scalar action in [-1, 1] to ppfd in [min_ppfd, max_ppfd]
        return await super().step(BALANCED_ACTION_100 * ppfd / 100)
