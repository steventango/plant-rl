from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber  # type: ignore
from environments.PlantGrowthChamber.utils import get_one_hot_time_observation


class CVPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_observation(self):  # type: ignore
        epoch_time, _, plant_stats = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)
        return get_one_hot_time_observation(local_time)
